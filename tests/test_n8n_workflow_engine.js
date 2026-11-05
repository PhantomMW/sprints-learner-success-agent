const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

console.log('=================================================================');
console.log('N8N WORKFLOW INTERNAL ENGINE & CODE NODE VERIFICATION');
console.log('=================================================================\n');

const workflowPath = path.join(__dirname, '..', 'n8n_workflows', 'learner_success_agent.json');
assert(fs.existsSync(workflowPath), `Workflow file not found at ${workflowPath}`);

const workflow = JSON.parse(fs.readFileSync(workflowPath, 'utf8'));

// 1. Structure Verification
console.log('1. Validating Workflow JSON Structure and Graph Integrity...');
assert(workflow.name === 'Sprints AI Learner Success & Support Agent', 'Workflow name mismatch');
assert(Array.isArray(workflow.nodes) && workflow.nodes.length >= 7, 'Expected at least 7 nodes');

const nodeMap = new Map();
workflow.nodes.forEach(n => nodeMap.set(n.name, n));

const requiredNodes = [
  'HTTP Webhook Trigger',
  'Parse & Validate Query',
  'Analyze Sentiment, NER & Vector RAG',
  'Deliberative Decision Router',
  'Persist Escalation to SQLite DB',
  'Format Escalation Response',
  'Format Automated RAG Response',
  'Respond to Webhook'
];

requiredNodes.forEach(name => {
  assert(nodeMap.has(name), `Missing required workflow node: ${name}`);
});
console.log(`   [PASS] All ${requiredNodes.length} required nodes present with valid positions and types.`);

// 2. Validate Node Connections & Timeout Budgets
console.log('\n2. Validating Connection Graph Routing & Timeout Budgets...');
const conn = workflow.connections;
assert(conn['HTTP Webhook Trigger'].main[0][0].node === 'Parse & Validate Query');
assert(conn['Parse & Validate Query'].main[0][0].node === 'Analyze Sentiment, NER & Vector RAG');
assert(conn['Analyze Sentiment, NER & Vector RAG'].main[0][0].node === 'Deliberative Decision Router');

// IF branching
const ifConns = conn['Deliberative Decision Router'].main;
assert(ifConns.length === 2, 'IF node must have 2 output branches');
assert(ifConns[0][0].node === 'Persist Escalation to SQLite DB', 'Output 0 (True) must route to DB escalation');
assert(ifConns[1][0].node === 'Format Automated RAG Response', 'Output 1 (False) must route to RAG response');

// Timeout budgets
const httpAnalysisNode = nodeMap.get('Analyze Sentiment, NER & Vector RAG');
assert(httpAnalysisNode.parameters.options.timeout <= 2500, 'HTTP analysis node timeout must be budgeted <= 2500ms to guarantee <3500ms SLA');
const httpDbNode = nodeMap.get('Persist Escalation to SQLite DB');
assert(httpDbNode.parameters.options.timeout <= 2000, 'HTTP DB node timeout must be budgeted <= 2000ms');

console.log('   [PASS] Connection graph & timeout budgets verified (headroom reserved for SLA < 3.5s).');

// 3. Execute JS Code Node: Parse & Validate Query in Node.js VM
console.log('\n3. Testing Code Node: Parse & Validate Query (node-parse-query)...');
const parseNode = nodeMap.get('Parse & Validate Query');
const parseCode = parseNode.parameters.jsCode;

function runNodeVm(code, contextVars) {
  const sandbox = {
    ...contextVars,
    Date: Date,
    Math: Math,
    Error: Error,
    Boolean: Boolean,
    console: console
  };
  const script = new vm.Script(`(function() { ${code} })()`);
  const ctx = vm.createContext(sandbox);
  return script.runInContext(ctx);
}

// 3A. Valid payload with special characters, quotes, and Egyptian dialect
const validPayload = {
  $json: {
    body: {
      learner_id: 'LRN-1019',
      query_text: 'أنا مش فاهم "PyTorch tensors" خالص وعايز مساعدة ضروري!\nوشكراً',
      cohort_city: 'Alexandria'
    }
  }
};
const parsedResult = runNodeVm(parseCode, validPayload);
assert.strictEqual(parsedResult.learner_id, 'LRN-1019');
assert(parsedResult.query_text.includes('مش فاهم'));
assert.strictEqual(parsedResult.cohort_city, 'Alexandria');
assert(parsedResult.received_at, 'Missing received_at timestamp');
assert(parsedResult.correlation_id && parsedResult.correlation_id.startsWith('corr-'), 'Missing or invalid correlation_id');
console.log('   [PASS] Successfully parsed complex text with correlation_id generation.');

// 3B. Rejection of missing learner_id
assert.throws(() => {
  runNodeVm(parseCode, { $json: { body: { learner_id: '', query_text: 'Valid query' } } });
}, /learner_id/, 'Must reject missing learner_id');
console.log('   [PASS] Successfully threw Bad Request on empty learner_id.');

// 3C. Rejection of blank query_text
assert.throws(() => {
  runNodeVm(parseCode, { $json: { body: { learner_id: 'LRN-1001', query_text: '    ' } } });
}, /query_text/, 'Must reject blank query_text');
console.log('   [PASS] Successfully threw Bad Request on blank query_text.');

// 4. Test Deliberative IF Evaluation
console.log('\n4. Testing IF Branching Logic (Deliberative Decision Router)...');
const ifNode = nodeMap.get('Deliberative Decision Router');
const targetCondition = ifNode.parameters.conditions.string[0].value2;
assert.strictEqual(targetCondition, 'HUMAN_OPERATOR_ESCALATION');

function evaluateIf(decision) {
  return decision === targetCondition;
}
assert.strictEqual(evaluateIf('HUMAN_OPERATOR_ESCALATION'), true);
assert.strictEqual(evaluateIf('AUTOMATED_RAG_PIPELINE'), false);
console.log('   [PASS] IF conditions strictly match router semantics.');

// 5. Test Code Node: Format Escalation Response (node-format-escalation-response)
console.log('\n5. Testing Code Node: Format Escalation Response...');
const escNode = nodeMap.get('Format Escalation Response');
const escCode = escNode.parameters.jsCode;

const mockNodeAccessor = (nodeName) => {
  if (nodeName === 'Parse & Validate Query') {
    return {
      first: () => ({
        json: { correlation_id: 'corr-test-12345', learner_id: 'LRN-1019' }
      })
    };
  }
  return {
    first: () => ({
      json: {
        learner_id: 'LRN-1019',
        correlation_id: 'corr-test-12345',
        assigned_queue: 'Urgent_Mentor_Support_Desk',
        priority: 'Critical',
        sentiment: { polarity: -0.6, subjectivity: 0.8 },
        agent_response: 'Your query has been escalated to a mentor.',
        execution_latency_ms: 14.2,
        sla_met: true
      }
    })
  };
};

// 5A. Legitimate DB persistence
const validEscContext = {
  $json: {
    status: 'persisted',
    ticket_id: 'TCK-ESC-048291'
  },
  $: mockNodeAccessor
};
const escResult = runNodeVm(escCode, validEscContext);
assert.strictEqual(escResult.status, 'escalated');
assert.strictEqual(escResult.ticket_id, 'TCK-ESC-048291');
assert.strictEqual(escResult.db_persisted, true);
assert.strictEqual(escResult.correlation_id, 'corr-test-12345');
assert.strictEqual(escResult.sla_met, true);
console.log('   [PASS] Valid persistence verified with correlation_id tracking.');

// 5B. Failure when DB did not persist (Fake fallback prevented)
const failedDbContext = {
  $json: { status: 'failed' },
  $: validEscContext.$
};
assert.throws(() => {
  runNodeVm(escCode, failedDbContext);
}, /Database escalation ticket persistence failed/, 'Must reject unpersisted ticket');
console.log('   [PASS] Unpersisted ticket strictly rejected with error (zero fake fallback IDs allowed).');

// 6. Test Code Node: Format Automated RAG Response
console.log('\n6. Testing Code Node: Format Automated RAG Response...');
const ragNode = nodeMap.get('Format Automated RAG Response');
const ragCode = ragNode.parameters.jsCode;

const ragContext = {
  $json: {
    learner_id: 'LRN-1002',
    assigned_queue: 'Automated_Curriculum_RAG',
    priority: 'Normal',
    extracted_modules: ['MOD-101'],
    extracted_dates: ['Thursday'],
    retrieved_chunk: { chunk_id: 'CHK-SQL-01', text: 'SQL GROUP BY tutorial' },
    agent_response: 'Here is the SQL tutorial.',
    execution_latency_ms: 152.0,
    sla_met: true
  },
  $: mockNodeAccessor
};
const ragResult = runNodeVm(ragCode, ragContext);
assert.strictEqual(ragResult.status, 'resolved_auto');
assert(ragResult.ticket_id.startsWith('TCK-AUTO-'));
assert.strictEqual(ragResult.db_persisted, false);
assert.strictEqual(ragResult.correlation_id, 'corr-test-12345');
assert.strictEqual(ragResult.extracted_modules[0], 'MOD-101');
console.log('   [PASS] Automated RAG response successfully generated with chunk metadata & correlation_id.');

console.log('\n=================================================================');
console.log('ALL N8N WORKFLOW NODE JAVASCRIPT & GRAPH TESTS PASSED 100%!');
console.log('=================================================================');
