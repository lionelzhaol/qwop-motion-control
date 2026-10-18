const Leaderboard = (() => {
  const boardBody = document.getElementById('board-body');
  const boardEmpty = document.getElementById('board-empty');
  const boardMode = document.getElementById('board-mode');
  const boardStatus = document.getElementById('board-status');
  const submitBtn = document.getElementById('submit-btn');

  const MODE_LABELS = { sprint: '100 米', hurdle: '跨栏', 'long-jump': '跳远' };
  const UNIT_LABELS = { frames: '时间', distance_mm: '距离' };

  var currentMode = null;
  var topMetrics = {};

  function formatMetric(value, unit) {
    if (unit === 'distance_mm') return (value / 1000).toFixed(3) + ' m';
    return (value / 30).toFixed(3) + ' s';
  }

  function escapeHtml(str) {
    var el = document.createElement('span');
    el.textContent = str;
    return el.innerHTML;
  }

  function renderRows(entries, unit) {
    boardBody.innerHTML = '';
    boardEmpty.style.display = entries.length ? 'none' : '';
    entries.forEach(function (entry, i) {
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + (i + 1) + '</td>' +
        '<td>' + escapeHtml(entry.player_name) + '</td>' +
        '<td class="col-result">' + formatMetric(entry.metric, unit) + '</td>';
      boardBody.appendChild(tr);
    });
  }

  async function load(mode) {
    var config = GameEngine.getModeConfig(mode);
    if (!config) return;
    var label = MODE_LABELS[mode] || mode;
    boardStatus.textContent = '正在加载排行榜…';
    try {
      var resp = await fetch('/api/rank/' + config.ruleset);
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      var data = await resp.json();
      var entries = data.entries || [];
      var unitLabel = UNIT_LABELS[data.metric_unit] || data.metric_unit;
      boardMode.textContent = label + ' · ' + unitLabel;
      topMetrics[mode] = entries.length ? entries[0].metric : null;
      renderRows(entries, data.metric_unit);
      boardStatus.textContent = '共 ' + entries.length + ' 条成绩';
      return true;
    } catch (err) {
      boardStatus.textContent = '排行榜加载失败';
      return false;
    }
  }

  function startPolling(mode) {
    currentMode = mode;
    load(mode);
  }

  function stopPolling() {}

  function refresh() {
    return currentMode ? load(currentMode) : Promise.resolve(false);
  }

  function isTopResult(mode, metric) {
    var topMetric = topMetrics[mode];
    if (topMetric === undefined) return false;
    if (topMetric === null) return true;
    return mode === 'long-jump' ? metric >= topMetric : metric <= topMetric;
  }

  function enableSubmit(enable) {
    submitBtn.disabled = !enable;
    submitBtn.title = enable ? '提交当前成绩到排行榜' : '完成比赛后才能提交';
  }

  async function submitReplay(playerName, replayBytes) {
    if (!replayBytes || !playerName) return;
    submitBtn.disabled = true;
    try {
      var form = new FormData();
      form.append('player_name', playerName);
      form.append('replay', new Blob([replayBytes]), 'replay.bin');
      var resp = await fetch('/api/replay', { method: 'POST', body: form });
      var data = await resp.json();
      if (!resp.ok) {
        UI.toast(data.error || '提交失败', true);
        return null;
      }
      if (data.qualified && !data.duplicate) {
        UI.toast((data.updated ? '已更新个人最佳！第 ' : '上榜！第 ') + data.rank + ' 名');
      } else if (data.duplicate) {
        UI.toast('重复提交，已在第 ' + data.rank + ' 名');
      } else if (data.reason === 'duplicate_metric') {
        UI.toast('排行榜中已有相同成绩');
      } else if (data.reason === 'not_personal_best') {
        UI.toast('该名字已有更好的成绩');
      } else {
        UI.toast('未进入 TOP 50');
      }
      return data;
    } catch (err) {
      UI.toast('网络错误', true);
      return null;
    } finally {
      submitBtn.disabled = true;
    }
  }

  return { load, startPolling, stopPolling, refresh, isTopResult, enableSubmit, submitReplay };
})();
