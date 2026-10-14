const MOVES = ['rock', 'paper', 'scissors'];
const MOVE_EMOJI = { rock: '✊', paper: '✋', scissors: '✌️' };
const MOVE_COLORS = { rock: '#ff9a57', paper: '#6ec5ff', scissors: '#a99aff' };
const STAGE_NAMES = ['Observed history', 'Feature engineering', 'Input layer', 'Hidden layer 1', 'Hidden layer 2', 'Output probabilities', 'Decision policy'];
let latestRunId = 0;
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const sleep = (ms) => new Promise(resolve => window.setTimeout(resolve, ms));

const game = {
  history: [],
  scores: { win: 0, tie: 0, loss: 0 },
  rounds: 0,
  aiMove: null,
  phase: 'initializing',
  latestTrace: null,
  inferenceCounter: 0,
};

function titleCase(value) { return value.charAt(0).toUpperCase() + value.slice(1); }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function fmt(value, places = 3) { return Number(value).toFixed(places); }
function setPhase(phase) {
  game.phase = phase;
  const canPlay = phase === 'ready';
  $$('.move-card').forEach(button => { button.disabled = !canPlay; });
  $('#replay-button').disabled = phase !== 'ready' || !game.latestTrace;
  $('#pipeline-state-dot').classList.toggle('running', phase === 'analyzing');
  $('#pipeline-state-text').textContent = phase === 'analyzing'
    ? 'FORWARD PASS RUNNING'
    : phase === 'ready'
      ? (game.latestTrace ? 'READY FOR INPUT' : 'AWAITING INPUT')
      : phase.toUpperCase();
  $('#pipeline-state-text').parentElement.classList.toggle('running', phase === 'analyzing');
}
function setResult(text, kind = '') {
  const node = $('#result-message');
  node.className = `result-message ${kind}`;
  node.lastElementChild.textContent = text;
}
function updateRecord() {
  $('#stat-record').innerHTML = `${game.scores.win} <span>W</span> · ${game.scores.tie} <span>T</span> · ${game.scores.loss} <span>L</span>`;
  $('#round-number').textContent = String(game.rounds + 1).padStart(2, '0');
}
function setAiCard(move, predictedPlayer, topProbability, round = game.rounds + 1, state = 'locked') {
  $('#ai-emoji').textContent = MOVE_EMOJI[move] || '◌';
  $('#ai-move-name').textContent = move ? titleCase(move) : (state === 'waiting' ? 'Awaiting input' : 'Analyzing…');
  $('#ai-move-sub').textContent = move
    ? `${state === 'previous' ? 'Previous round' : 'Action selected'} · predicted ${titleCase(predictedPlayer)} (${Math.round(topProbability * 100)}%)`
    : (state === 'waiting' ? 'Your selection triggers inference' : 'Features → neural layers → policy');
  $('#ai-status').textContent = move
    ? (state === 'previous' ? `Round ${String(round).padStart(2, '0')} resolved` : `Action locked for round ${String(round).padStart(2, '0')}`)
    : (state === 'waiting' ? 'Ready for your move' : 'Forward pass running');
}
function formatMoveToken(move, index, total) {
  return `<span class="history-token ${index === total - 1 ? 'latest' : ''}" title="Round ${index + 1}">${MOVE_EMOJI[move]} ${titleCase(move)}</span>`;
}
function smallKV(label, value) {
  return `<div class="small-kv"><span>${label}</span><strong>${value}</strong></div>`;
}
function renderHistoryStage(trace) {
  const recent = trace.recent_history;
  $('#history-size').textContent = `${trace.history_length} moves`;
  const tokens = recent.length
    ? `<div class="history-tokens">${recent.map((move, i) => formatMoveToken(move, i, recent.length)).join('')}</div>`
    : '<div class="placeholder">No prior moves yet. The network starts from its learned prior.</div>';
  const frequencies = MOVES.map(move => {
    const count = trace.frequency_counts[move] || 0;
    return smallKV(titleCase(move), `${count} · ${trace.feature_window_length ? Math.round(count / trace.feature_window_length * 100) : 0}%`);
  }).join('');
  $('#stage-history').innerHTML = `${tokens}<div style="margin-top:9px">${frequencies}</div>`;
}
function renderFeatureStage(trace) {
  const groups = [
    ['RECENT MOVES', '15 one-hot values'],
    ['MOVE FREQUENCIES', '3 proportions'],
    ['TRANSITION MATRIX', '9 smoothed values'],
    ['HISTORY LENGTH', '1 scaled value'],
  ];
  $('#stage-features').innerHTML = groups.map(([name, description]) =>
    `<div class="small-kv"><span>${name}</span><strong>${description}</strong></div>`
  ).join('') + `<div class="layer-caption"><span>VECTOR BUILT</span><strong>${trace.features.length} × 1</strong></div>`;
}
function renderNeuronStage(containerId, values, kind, countLabel) {
  const n = values.length;
  const maxAbs = Math.max(0.00001, ...values.map(value => Math.abs(value)));
  const gridClass = kind === 'input' ? 'input' : kind === 'hidden48' ? 'hidden48' : kind === 'hidden24' ? 'hidden24' : '';
  const cells = values.map((value, i) => {
    const absRatio = Math.abs(value) / maxAbs;
    const opacity = (0.12 + absRatio * 0.88).toFixed(2);
    const typeClass = value < 0 ? 'negative' : '';
    return `<span class="neuron ${typeClass}" style="opacity:${opacity}" title="Neuron ${i + 1}: ${fmt(value, 5)}"></span>`;
  }).join('');
  const min = Math.min(...values), max = Math.max(...values), active = values.filter(value => value > 0).length;
  $(`#${containerId}`).innerHTML = `<div class="neuron-grid ${gridClass}">${cells}</div><div class="layer-caption"><span>${active} positive</span><span>min ${fmt(min, 2)} · max ${fmt(max, 2)}</span></div>`;
}
function renderOutputStage(trace) {
  const list = MOVES.map(move => {
    const p = trace.probabilities[move];
    return `<div class="prob-row"><span class="prob-label">${titleCase(move)}</span><div class="prob-track"><div class="prob-fill ${move}" style="width:${(p * 100).toFixed(2)}%"></div></div><span class="prob-value">${(p * 100).toFixed(1)}%</span></div>`;
  }).join('');
  const logits = trace.layers[trace.layers.length - 1].pre_activation || [];
  const logitSummary = MOVES.map((move, i) => `${move[0].toUpperCase()}:${fmt(logits[i] ?? 0, 2)}`).join(' · ');
  $('#stage-output').innerHTML = `<div class="prob-stack">${list}</div><div class="layer-caption"><span>LOGITS</span><strong>${logitSummary}</strong></div><div class="layer-caption"><span>PROBABILITY SUM</span><strong>${fmt(Object.values(trace.probabilities).reduce((a,b)=>a+b,0), 4)}</strong></div>`;
}
function renderDecisionStage(trace) {
  const maxUtil = Math.max(0.001, ...trace.utilities.map(item => Math.abs(item.expected_payoff)));
  const rows = trace.utilities.map(item => {
    const width = Math.max(3, Math.abs(item.expected_payoff) / maxUtil * 100);
    const colorClass = item.expected_payoff > 0.0001 ? 'positive' : item.expected_payoff < -0.0001 ? 'negative' : '';
    return `<div class="utility-row ${item.move === trace.selected_ai_move ? 'chosen' : ''}"><span class="utility-name">${titleCase(item.move)}</span><div class="utility-track"><div class="utility-fill ${colorClass}" style="width:${width}%"></div></div><span class="utility-value">${item.expected_payoff >= 0 ? '+' : ''}${fmt(item.expected_payoff, 3)}</span></div>`;
  }).join('');
  $('#stage-decision').innerHTML = `${rows}<span class="chosen-tag">↗ SELECT ${trace.selected_ai_move.toUpperCase()}</span>`;
}
function renderFeatureTable(trace) {
  $('#feature-table').innerHTML = trace.features.map(feature =>
    `<tr title="${feature.name}: raw ${fmt(feature.raw, 5)}, standardized ${fmt(feature.scaled, 5)}"><td>${feature.name}</td><td>${fmt(feature.raw, 4)}</td><td>${fmt(feature.scaled, 4)}</td></tr>`
  ).join('');
  $('#feature-table-footer').textContent = `${trace.features.length} engineered signals · StandardScaler fitted on training data`;
}
function renderProbabilityDetail(trace) {
  $('#probability-detail').innerHTML = MOVES.map(move => {
    const p = trace.probabilities[move];
    return `<div class="probability-card-row"><span class="probability-card-name">${MOVE_EMOJI[move]} ${titleCase(move)}</span><div class="probability-card-track"><div class="probability-card-fill" style="width:${p*100}%;background:${MOVE_COLORS[move]}"></div></div><span class="probability-card-pct">${(p*100).toFixed(2)}%</span></div>`;
  }).join('');
  const predicted = trace.predicted_player_move;
  $('#decision-summary').innerHTML = `<div class="summary-label">NEXT-MOVE PREDICTION</div><div class="summary-value">${MOVE_EMOJI[predicted]} ${titleCase(predicted)} <span style="font:10px var(--mono);color:#9aabc4">· ${(trace.probabilities[predicted]*100).toFixed(1)}%</span></div><div class="section-copy" style="margin-top:6px">This is a probability estimate from prior rounds, not a certainty.</div>`;
  $('#utility-list').innerHTML = '<div class="section-kicker" style="margin:13px 0 5px">EXPECTED PAYOFF PER ACTION</div>' + trace.utilities.map(item => {
    const isChosen = item.move === trace.selected_ai_move;
    return `<div class="utility-row ${isChosen ? 'chosen' : ''}"><span class="utility-name">${titleCase(item.move)}${isChosen ? ' ↗' : ''}</span><div class="utility-track"><div class="utility-fill ${item.expected_payoff > 0 ? 'positive' : item.expected_payoff < 0 ? 'negative' : ''}" style="width:${Math.max(3, Math.abs(item.expected_payoff)*100)}%"></div></div><span class="utility-value">${item.expected_payoff >= 0 ? '+' : ''}${fmt(item.expected_payoff)}</span></div>`;
  }).join('');
}
function renderTrace(trace, isReplay = false) {
  game.latestTrace = trace;
  renderHistoryStage(trace);
  renderFeatureStage(trace);
  renderNeuronStage('stage-input', trace.layers[0].values, 'input');
  renderNeuronStage('stage-hidden1', trace.layers[1].values, 'hidden48');
  renderNeuronStage('stage-hidden2', trace.layers[2].values, 'hidden24');
  renderOutputStage(trace);
  renderDecisionStage(trace);
  renderFeatureTable(trace);
  renderProbabilityDetail(trace);
  if (!isReplay) $('#inference-id').textContent = `INF-${String(++game.inferenceCounter).padStart(4,'0')}`;
  const err = trace.forward_pass_max_abs_error;
  $('#forward-check').textContent = err !== null && err < 1e-6 ? `MATCH · ${err.toExponential(1)}` : err !== null ? `Δ ${err.toExponential(1)}` : 'SCIKIT-LEARN';
  $('#trace-explanation').textContent = trace.history_length === 0
    ? 'No completed rounds are available yet. The ANN applies its learned synthetic-data prior to estimate this round’s move. Your currently locked input is excluded from the vector; after the round, it becomes history for the next inference.'
    : `Used ${trace.history_length} completed move${trace.history_length===1?'':'s'} (the latest ${trace.recent_history.length} are shown). The move you just submitted is excluded from this vector. The ANN estimates that move from prior history, then the policy compares expected payoff for all three AI responses.`;
}
function setStageState(index, status) {
  const stage = $(`.stage-card[data-stage="${index}"]`);
  stage.classList.toggle('is-active', status === 'active');
  stage.classList.toggle('is-complete', status === 'complete');
  stage.querySelector('.stage-status').textContent = status === 'active' ? 'PROCESSING' : status === 'complete' ? 'COMPLETE' : 'WAITING';
}
async function animatePipeline(trace, runId) {
  const packet = $('#data-packet');
  packet.classList.add('visible');
  $$('.stage-card').forEach((_, index) => setStageState(index, 'waiting'));
  $('#pipeline-progress').style.width = '0%';
  for (let index = 0; index < STAGE_NAMES.length; index++) {
    if (runId !== latestRunId) return false;
    setStageState(index, 'active');
    const width = ((index + 0.5) / STAGE_NAMES.length) * 100;
    $('#pipeline-progress').style.width = `${width}%`;
    packet.style.left = `${width}%`;
    await sleep(window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 40 : 560);
    if (runId !== latestRunId) return false;
    setStageState(index, 'complete');
  }
  $('#pipeline-progress').style.width = '100%';
  packet.style.left = '100%';
  await sleep(120);
  packet.classList.remove('visible');
  return true;
}
async function runAnalysis(historySnapshot, pendingPlayerMove, isReplay = false) {
  const runId = ++latestRunId;
  setPhase('analyzing');
  game.aiMove = null;
  setAiCard(null, null, 0);
  $('#player-status').textContent = isReplay
    ? 'Replaying the previous round’s trace…'
    : `Input locked: ${titleCase(pendingPlayerMove)}. The current move is excluded; only earlier rounds enter the ANN.`;
  $('#ai-move-name').textContent = 'Analyzing…';
  $('#ai-move-sub').textContent = 'Features → neural layers → policy';
  setResult(isReplay
    ? 'Replaying a completed round. No new move will be scored.'
    : `Your ${titleCase(pendingPlayerMove)} input is locked. Watch the ANN analyze prior history before its action is revealed.`, '');
  try {
    let trace;
    if (isReplay) {
      trace = game.latestTrace;
      if (!trace) throw new Error('There is no completed inference to replay yet.');
    } else {
      const response = await fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history: historySnapshot }),
      });
      trace = await response.json();
      if (runId !== latestRunId) return;
      if (!response.ok) throw new Error(trace.error || `Inference failed (${response.status})`);
    }
    renderTrace(trace, isReplay);
    const animationCompleted = await animatePipeline(trace, runId);
    if (!animationCompleted || runId !== latestRunId) return;

    if (isReplay) {
      game.aiMove = trace.selected_ai_move;
      setAiCard(game.aiMove, trace.predicted_player_move, trace.probabilities[trace.predicted_player_move], game.rounds, 'previous');
      setPhase('ready');
      $('#player-status').textContent = 'Replay complete. Choose a new move whenever you are ready.';
      setResult('Previous inference replayed. No new round was added to the record.', '');
      return;
    }

    // This action is chosen from the pre-input history snapshot. The new player
    // input is not included in the model request, so the current label cannot leak.
    const selectedAiMove = trace.selected_ai_move;
    const predicted = trace.predicted_player_move;
    const confidence = trace.probabilities[predicted];
    game.aiMove = selectedAiMove;
    setAiCard(selectedAiMove, predicted, confidence, game.rounds + 1, 'locked');

    const outcome = compare(pendingPlayerMove, selectedAiMove);
    game.rounds += 1;
    game.scores[outcome] += 1;
    game.history = [...historySnapshot, pendingPlayerMove];
    updateRecord();
    $$('.move-card').forEach(button => button.classList.toggle('selected', button.dataset.move === pendingPlayerMove));
    const descriptions = {
      win: `You win this round — ${titleCase(pendingPlayerMove)} beats ${titleCase(selectedAiMove)}.`,
      tie: `A tie — you both played ${titleCase(pendingPlayerMove)}.`,
      loss: `ANN wins this round — ${titleCase(selectedAiMove)} beats ${titleCase(pendingPlayerMove)}.`,
    };
    setAiCard(selectedAiMove, predicted, confidence, game.rounds, 'previous');
    $('#player-status').textContent = `Logged input: ${titleCase(pendingPlayerMove)} · ${outcome.toUpperCase()}`;
    setResult(`${descriptions[outcome]} Your move is now available as history for the next round.`, outcome);
    setPhase('ready');
  } catch (error) {
    if (runId !== latestRunId) return;
    setPhase('error');
    $('#player-status').textContent = 'Could not run inference.';
    $('#ai-move-name').textContent = 'Model unavailable';
    $('#ai-move-sub').textContent = error.message;
    setResult(`Error: ${error.message}. Ensure dependencies are installed and the model has been trained.`, 'loss');
    console.error(error);
  }
}
function compare(player, ai) {
  if (player === ai) return 'tie';
  return (player === 'rock' && ai === 'scissors') ||
    (player === 'paper' && ai === 'rock') ||
    (player === 'scissors' && ai === 'paper') ? 'win' : 'loss';
}
async function chooseMove(move) {
  if (game.phase !== 'ready') return;
  const historySnapshot = [...game.history];
  $$('.move-card').forEach(button => button.classList.toggle('selected', button.dataset.move === move));
  await runAnalysis(historySnapshot, move, false);
}
async function replayLastPipeline() {
  if (game.phase !== 'ready' || !game.latestTrace) return;
  await runAnalysis([...game.history], null, true);
}
function clearTraceVisualization() {
  $$('.stage-card').forEach((_, index) => setStageState(index, 'waiting'));
  $('#pipeline-progress').style.width = '0%';
  $('#data-packet').classList.remove('visible');
  $('#stage-history').innerHTML = '<div class="placeholder">Your first move will trigger inference…</div>';
  $('#stage-features').innerHTML = '<div class="placeholder">Building 28 features…</div>';
  $('#stage-input').innerHTML = '<div class="placeholder">Awaiting vector…</div>';
  $('#stage-hidden1').innerHTML = '<div class="placeholder">Computing activations…</div>';
  $('#stage-hidden2').innerHTML = '<div class="placeholder">Computing activations…</div>';
  $('#stage-output').innerHTML = '<div class="placeholder">Scoring 3 classes…</div>';
  $('#stage-decision').innerHTML = '<div class="placeholder">Comparing actions…</div>';
  $('#feature-table').innerHTML = '<tr><td colspan="3" class="empty-cell">Run inference to inspect the feature vector.</td></tr>';
  $('#probability-detail').innerHTML = '<div class="placeholder">Output probabilities will appear here.</div>';
  $('#decision-summary').innerHTML = '<div class="summary-label">NEXT-MOVE PREDICTION</div><div class="summary-value">Waiting for inference</div>';
  $('#utility-list').innerHTML = '';
  $('#inference-id').textContent = '—';
  $('#forward-check').textContent = '—';
  $('#trace-explanation').textContent = 'Choose a move to start. The current input is locked while the model analyzes only completed prior rounds, so the current move is not leaked into its prediction.';
}
function showError(message) {
  let banner = $('#fatal-error');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'fatal-error'; banner.className = 'error-banner';
    $('.intro').after(banner);
  }
  banner.textContent = message;
}
async function loadMetrics() {
  try {
    const response = await fetch('/api/metrics');
    const metrics = await response.json();
    if (!response.ok) throw new Error(metrics.error || 'Training metrics not found');
    $('#stat-samples').textContent = Number(metrics.dataset_samples).toLocaleString();
    $('#stat-accuracy').textContent = `${(metrics.test_accuracy * 100).toFixed(1)}%`;
  } catch (error) {
    $('#stat-samples').textContent = 'Train first';
    $('#stat-accuracy').textContent = '—';
  }
}
function resetGame() {
  ++latestRunId; // Cancel any in-progress animation or network response.
  game.history = [];
  game.scores = { win: 0, tie: 0, loss: 0 };
  game.rounds = 0;
  game.aiMove = null;
  game.latestTrace = null;
  $$('.move-card').forEach(button => button.classList.remove('selected'));
  updateRecord();
  clearTraceVisualization();
  setAiCard(null, null, 0, 1, 'waiting');
  $('#player-status').textContent = 'Choose Rock, Paper, or Scissors to start the live trace.';
  setResult('Make a move to stream the input through the real ANN. The current selection will not be used as a feature for that same round.', '');
  setPhase('ready');
}

$$('.move-card').forEach(button => button.addEventListener('click', () => chooseMove(button.dataset.move)));
$('#reset-button').addEventListener('click', resetGame);
$('#replay-button').addEventListener('click', replayLastPipeline);

(async function init() {
  await loadMetrics();
  try {
    const healthResponse = await fetch('/api/health');
    const health = await healthResponse.json();
    if (!health.ok) {
      showError(health.error || 'Model could not be loaded. Generate the dataset and train the ANN before starting Flask.');
      setPhase('error');
      $('#ai-move-name').textContent = 'Model unavailable';
      $('#ai-move-sub').textContent = 'Train model to begin';
      return;
    }
  } catch (error) {
    showError('Could not contact the Flask server. Start the application with `python app.py` and refresh this page.');
    setPhase('error');
    return;
  }
  game.aiMove = null;
  setAiCard(null, null, 0, 1, 'waiting');
  $('#player-status').textContent = 'Choose Rock, Paper, or Scissors to start the live trace.';
  setResult('Make a move to stream the input through the real ANN. Your current selection is excluded from that round’s model features.', '');
  setPhase('ready');
})();
