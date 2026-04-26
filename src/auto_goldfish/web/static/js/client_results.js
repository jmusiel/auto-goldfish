/**
 * Client-side results renderer.
 *
 * Renders simulation results from JSON into HTML, replicating the
 * server-side results_content.html template. Used when simulations
 * run client-side via Pyodide.
 */

const ClientResults = (function() {
    'use strict';

    // -- Tooltip management (shared with server-side rendering) --

    let tooltip = null;
    const tooltipCache = {};

    function ensureTooltip() {
        tooltip = document.getElementById('card-preview-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'card-preview-tooltip';
            document.body.appendChild(tooltip);
        }
    }

    function positionTooltip(e) {
        const x = e.clientX + 15;
        const y = Math.max(10, e.clientY - 180);
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
    }

    function rebindTooltips() {
        ensureTooltip();
        document.querySelectorAll('.card-link').forEach(link => {
            if (link._tooltipBound) return;
            link._tooltipBound = true;
            link.addEventListener('mouseenter', function(e) {
                const name = this.dataset.cardName;
                if (!tooltipCache[name]) {
                    const img = document.createElement('img');
                    img.src = 'https://api.scryfall.com/cards/named?exact='
                        + encodeURIComponent(name) + '&format=image&version=normal';
                    img.alt = name;
                    tooltipCache[name] = img;
                }
                tooltip.innerHTML = '';
                tooltip.appendChild(tooltipCache[name]);
                tooltip.style.display = 'block';
                positionTooltip(e);
            });
            link.addEventListener('mousemove', positionTooltip);
            link.addEventListener('mouseleave', function() {
                tooltip.style.display = 'none';
            });
        });
    }

    // -- HTML generation helpers --

    function fmt(val, decimals) {
        return Number(val).toFixed(decimals);
    }

    function cardLink(name) {
        return '<a class="card-link" data-card-name="' + escapeHtml(name)
            + '" href="https://scryfall.com/search?exact='
            + encodeURIComponent(name) + '" target="_blank">' + escapeHtml(name) + '</a>';
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function formatConfig(str) {
        // Render (mv2) portions as <sub> for compact display
        return escapeHtml(str).replace(/\(([^)]+)\)/g, '<sub>($1)</sub>');
    }

    // -- Section renderers --

    // Per-stat copy used by the CASTER Score header tooltip + explanation panel.
    const CASTER_STATS = [
        {name: 'Consistency', key: 'consistency', color: '#eab308',
         desc: 'How rarely the deck bricks',
         lowMeaning: 'worst-case games are catastrophic — frequent floods, stalls, or dead hands.',
         highMeaning: 'almost every game runs smoothly; the bottom 25% looks much like the average.'},
        {name: 'Acceleration', key: 'acceleration', color: '#ef4444',
         desc: 'Early-game mana deployment',
         lowMeaning: 'slow opening — barely any mana spent over the first four turns.',
         highMeaning: 'explosive early game with multiple ramp/draw plays before turn 5.'},
        {name: 'Snowball', key: 'snowball', color: '#8b5cf6',
         desc: 'How much advantage compounds over time',
         lowMeaning: 'late game stays flat — the deck does not pull away from its early curve.',
         highMeaning: 'late turns dwarf the early game; the deck snowballs hard once it gets going.'},
        {name: 'Toughness', key: 'toughness', color: '#22c55e',
         desc: 'Structural redundancy of the decklist',
         lowMeaning: 'thin/fragile decklist — few mana sources, little draw, or a heavy curve.',
         highMeaning: 'deep redundancy: lots of mana sources, draw, low-cost plays, and a controlled curve.'},
        {name: 'Efficiency', key: 'efficiency', color: '#3b82f6',
         desc: 'Mana utilization per turn',
         lowMeaning: 'mana left unused — frequent mid-game stall turns.',
         highMeaning: 'nearly every turn fully utilized; very few wasted mana points.'},
        {name: 'Reach', key: 'reach', color: '#f97316',
         desc: 'Peak mana output and ceiling',
         lowMeaning: 'low ceiling — even the best games never spend that much mana.',
         highMeaning: 'explosive peak turns; the top 25% of games spend a huge amount of mana.'},
    ];

    function renderCasterExplanation() {
        const cal = window.CASTER_CALIBRATION || {};

        let html = '<details class="caster-help">';
        html += '<summary>What does my CASTER Score mean?</summary>';
        html += '<div class="caster-help-body">';
        html += '<p>Each stat is rescaled to <strong>1&ndash;10</strong> against ';
        if (cal.calibrated && cal.n_rows > 0) {
            html += 'an empirical distribution of <strong>' + cal.n_rows + ' run'
                + (cal.n_rows === 1 ? '' : 's') + '</strong> across <strong>'
                + cal.n_decks + ' deck' + (cal.n_decks === 1 ? '' : 's') + '</strong> '
                + 'in this database (anchors at p' + Math.round(cal.low_pct) + '/p'
                + Math.round(cal.high_pct) + ' with Bayesian shrinkage, pseudo_count='
                + cal.pseudo_count + '). ';
        } else {
            html += 'the built-in default anchors (no DB calibration available). ';
        }
        html += 'A score of <strong>1</strong> matches the low anchor; <strong>10</strong> matches the high anchor.</p>';

        html += '<table class="caster-help-table"><thead><tr>'
            + '<th>Stat</th><th>What it measures</th>'
            + '<th>Score 1 looks like…</th><th>Score 10 looks like…</th>'
            + '</tr></thead><tbody>';
        for (const s of CASTER_STATS) {
            html += '<tr>';
            html += '<td><strong style="color:' + s.color + '">' + s.name + '</strong></td>';
            html += '<td>' + escapeHtml(s.desc) + '</td>';
            html += '<td>' + escapeHtml(s.lowMeaning) + '</td>';
            html += '<td>' + escapeHtml(s.highMeaning) + '</td>';
            html += '</tr>';
        }
        html += '</tbody></table>';
        if (!cal.calibrated) {
            html += '<p class="caster-help-foot">Set <code>AUTO_GOLDFISH_CALIBRATE=1</code> '
                + '(default) and run more decks to switch from default anchors to a live calibration.</p>';
        } else {
            html += '<p class="caster-help-foot">Calibration refreshes automatically as new runs land in the DB. '
                + 'Set <code>AUTO_GOLDFISH_CALIBRATE=0</code> to disable.</p>';
        }
        html += '</div></details>';
        return html;
    }

    function renderDeckScore(results) {
        // Use the last result (highest land count) for the score
        const r = results[results.length - 1];
        const score = r.deck_score;
        if (!score) return '';

        let html = '<div class="deck-score-section"><h2>CASTER Score</h2>';
        html += renderCasterExplanation();
        html += '<div class="deck-score-grid">';
        // Radar chart canvas
        html += '<div class="deck-score-radar"><canvas id="deckScoreRadar"></canvas></div>';
        // Stat bars
        html += '<div class="deck-score-bars">';
        for (const s of CASTER_STATS) {
            const val = score[s.key] || 0;
            const pct = (val / 10 * 100).toFixed(0);
            html += '<div class="deck-stat-row">';
            html += '<div class="deck-stat-label" data-tip="' + escapeHtml(s.desc) + '">' + s.name + '</div>';
            html += '<div class="deck-stat-bar-track">';
            html += '<div class="deck-stat-bar-fill" style="width:' + pct + '%;background:' + s.color + '"></div>';
            html += '</div>';
            html += '<div class="deck-stat-value">' + val + '</div>';
            html += '</div>';
        }
        html += '</div></div></div>';
        return html;
    }

    function renderDeckScoreChart(results) {
        const r = results[results.length - 1];
        const score = r.deck_score;
        if (!score) return;

        const canvas = document.getElementById('deckScoreRadar');
        if (!canvas) return;

        const labels = ['Consistency', 'Acceleration', 'Snowball', 'Toughness', 'Efficiency', 'Reach'];
        const values = [score.consistency, score.acceleration, score.snowball, score.toughness, score.efficiency, score.reach];

        new Chart(canvas, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'CASTER Score',
                    data: values,
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointBackgroundColor: '#3b82f6',
                    pointRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        min: 0,
                        max: 10,
                        ticks: { stepSize: 2, display: true },
                        pointLabels: { font: { size: 13, weight: 'bold' } },
                    }
                },
                plugins: {
                    legend: { display: false },
                }
            }
        });
    }

    function renderSummaryTable(results, isOptimization) {
        let html = '<h2>' + (isOptimization ? 'Optimization Results' : 'Summary Statistics') + '</h2>';
        if (isOptimization) {
            html += '<p class="hint">Ranked by optimization target. Top configurations evaluated with full simulation count.</p>';
        }
        html += `<details class="metric-descriptions">
            <summary>Metric Definitions</summary>
            <dl class="metric-list">
                <dt>Consistency</dt>
                <dd>Left-tail ratio: mean mana in the worst 25% of games divided by the overall mean (0&ndash;1 scale). 1.0 = perfectly consistent; lower values mean bad games are much worse than average. Based on the selected mana mode.</dd>
                <dt>Avg Spells</dt>
                <dd>Average number of spells cast per game.</dd>
                <dt>Mana Spent: V+D</dt>
                <dd>Total mana spent on value (no-effect) and draw spells. Ramp excluded because it pays for itself. Higher = more resources deployed.</dd>
                <dt>Mana Spent: Value / Draw / Ramp</dt>
                <dd>Mana breakdown by card type. Draw > ramp priority (cards with both count as draw).</dd>
                <dt>Mana Spent: All</dt>
                <dd>Total mana spent on all spells (value + draw + ramp).</dd>
                <dt>Hand Sum</dt>
                <dd>Sum of min(hand_size, 7) per turn. Measures card availability across the game.</dd>
                <dt>Bad Turns</dt>
                <dd>Average turns where no spells were cast and the deck wasn&rsquo;t empty. Lower = better.</dd>
                <dt>Mid Turns</dt>
                <dd>Average turns where fewer than 2 spells were cast, mana spent was below the turn number, and the deck wasn&rsquo;t empty. Lower = better.</dd>
                <dt>Avg Lands / Avg Mulls</dt>
                <dd>Average lands played and mulligans taken per game.</dd>
                <dt>Avg Draws</dt>
                <dd>Average cards drawn per game.</dd>
                <dt>Mana Percentiles (25th / 50th / 75th)</dt>
                <dd>Percentiles of mana spent (based on selected mana mode) showing distribution spread.</dd>
            </dl>
        </details>`;
        html += '<div class="table-wrap"><table class="stats-table"><thead><tr>';
        if (isOptimization) html += '<th rowspan="2">Rank</th><th rowspan="2">Configuration</th>';
        html += '<th rowspan="2">Lands</th><th rowspan="2">Consistency</th><th rowspan="2">Avg Spells</th>';
        html += '<th colspan="5">Mana Spent <span class="mana-view-toggle" data-current="bottom_25">';
        html += '<button class="mana-view-btn active" data-view="bottom_25" title="Average mana in worst 25% of games">Floor</button>';
        html += '<button class="mana-view-btn" data-view="mean" title="Average mana across all games">Mean</button>';
        html += '<button class="mana-view-btn" data-view="top_25" title="Average mana in best 25% of games">Ceiling</button>';
        html += '</span></th>';
        html += '<th rowspan="2">Hand Sum</th><th rowspan="2">Bad Turns</th>';
        html += '<th rowspan="2">Mid Turns</th><th rowspan="2">Avg Lands</th><th rowspan="2">Avg Mulls</th>';
        html += '<th rowspan="2">Avg Draws</th>';
        html += '<th colspan="3">Mana Percentiles</th></tr><tr>';
        html += '<th>Value</th><th>Draw</th><th>Ramp</th><th>V+D</th><th>All</th>';
        html += '<th>25th</th><th>50th</th><th>75th</th></tr></thead><tbody>';

        const colCount = isOptimization ? 19 : 17;
        for (let i = 0; i < results.length; i++) {
            const r = results[i];
            const isBaselineRef = isOptimization && r.opt_baseline_rank != null;

            // Insert separator before the baseline reference row
            if (isBaselineRef) {
                html += '<tr class="baseline-separator"><td colspan="' + colCount + '"></td></tr>';
            }

            const conMargin = r.ci_consistency ? (r.ci_consistency[1] - r.ci_consistency[0]) / 2 : 0;
            let rowStyle = '';
            if (isOptimization && i === 0) rowStyle = ' style="font-weight:bold; background:#e8f5e9;"';
            if (isBaselineRef) rowStyle = ' style="opacity:0.75; border-top:2px dashed #94a3b8;"';
            html += '<tr' + rowStyle + '>';
            if (isOptimization) {
                const rank = isBaselineRef ? r.opt_baseline_rank : (i + 1);
                html += '<td>' + rank + '</td>';
                const sourceBadge = r.opt_source ? ' <span class="source-badge source-' + r.opt_source + '">' + r.opt_source + '</span>' : '';
                html += '<td style="text-align:left">' + formatConfig(r.opt_config || 'Base deck') + sourceBadge + '</td>';
            }
            html += '<td>' + r.land_count + '</td>';
            html += '<td>' + fmt(r.consistency, 3) + ' <small>&plusmn;' + fmt(conMargin, 4) + '</small></td>';
            html += '<td>' + fmt(r.mean_spells_cast ?? 0, 2) + '</td>';
            var qm = r.quartile_mana || {};
            var bot = qm.bottom_25 || {}; var mn = qm.mean || {}; var top = qm.top_25 || {};
            var manaFields = ['value', 'draw', 'ramp', 'vd', 'all'];
            var ciFields = [r.ci_mana_value ?? 0, r.ci_mana_draw ?? 0, r.ci_mana_ramp ?? 0, r.ci_mana ?? 0, r.ci_mana_total ?? 0];
            for (var mi = 0; mi < manaFields.length; mi++) {
                var fld = manaFields[mi];
                var bv = bot[fld] ?? 0, mv = mn[fld] ?? 0, tv = top[fld] ?? 0;
                html += '<td class="mana-cell" data-bot="' + fmt(bv, 2) + '" data-mean="' + fmt(mv, 2) + '" data-top="' + fmt(tv, 2)
                    + '" data-ci="' + fmt(ciFields[mi], 2) + '">'
                    + fmt(bv, 2) + ' <small>&plusmn;' + fmt(ciFields[mi], 2) + '</small></td>';
            }
            html += '<td>' + fmt(r.mean_hand_sum ?? 0, 1) + '</td>';
            html += '<td>' + fmt(r.mean_bad_turns, 2) + '</td>';
            html += '<td>' + fmt(r.mean_mid_turns, 2) + '</td>';
            html += '<td>' + fmt(r.mean_lands, 2) + '</td>';
            html += '<td>' + fmt(r.mean_mulls, 2) + '</td>';
            html += '<td>' + fmt(r.mean_draws ?? 0, 2) + '</td>';
            html += '<td>' + fmt(r.percentile_25, 1) + '</td>';
            html += '<td>' + fmt(r.percentile_50, 1) + '</td>';
            html += '<td>' + fmt(r.percentile_75, 1) + '</td>';
            html += '</tr>';
        }
        html += '</tbody></table></div>';
        return html;
    }

    const ORDINALS = ['', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'];
    function ordinal(k) { return ORDINALS[k] || (k + 'th'); }

    function renderCardPerformance(results) {
        const cp = results[0].card_performance;
        if (!cp || !cp.high_performing) return '';

        let html = '<h2>Card Performance</h2>';
        html += '<details class="card-perf-help">'
            + '<summary>How to read this table</summary>'
            + '<p>Spells with the same mana cost and same simulator-relevant effects are pooled into a single archetype, since they\'re interchangeable to the simulator. The example shown is just one card from the pool.</p>'
            + '<ul class="card-perf-help-list">'
            + '<li><strong>Impact</strong> — Average mana spent in games where you drew at least one copy of this spell, minus games where you didn\'t. Positive = the deck does more when this is in your opening hand or draws.</li>'
            + '<li><strong>Each extra copy</strong> — How much the 1st, 2nd, 3rd… copy adds on top of the previous count. "+0.50" on the 2nd means having two in hand spends 0.50 more mana than just one. A faded "—" means the 90% CI overlaps zero (effect indistinguishable from chance); ordinals the deck rarely realizes are not shown at all.</li>'
            + '<li><strong>Recommendation</strong> — A plain-English read of the per-copy trend: add more, you have enough, cut copies, or there isn\'t enough data.</li>'
            + '<li><sup class="always-drawn">∑</sup> means the spell appeared in nearly every game (so there is no "drew none" comparison group). The Impact shown is the sum of the per-copy effects that were statistically significant.</li>'
            + '</ul>'
            + '<p><em>Lands are excluded, except for MDFCs (cards with both a land face and a spell face).</em></p>'
            + '</details>';
        const supplementalSuffix = cp.supplemental_games
            ? ' (+' + cp.supplemental_games + ' supplemental games for marginal precision)'
            : '';
        html += '<p class="card-perf-summary">Impact of drawing each spell on average mana spent across '
            + cp.total_games + ' games' + supplementalSuffix + '.</p>';
        html += '<div class="card-perf-grid">';

        function renderMarginals(card) {
            if (!card.marginals || !card.marginals.length) return '';
            const parts = card.marginals.map(m => {
                const ord = ordinal(m.k);
                if (m.noise) {
                    const tip = ord + ' copy: 90% CI overlaps zero, so the effect is too small to distinguish from chance ('
                        + m.n_curr + ' games drew exactly ' + m.k + ').';
                    return '<span class="marginal noise" data-tip="' + escapeHtml(tip) + '">'
                        + ord + ': —</span>';
                }
                const cls = m.effect > 0 ? 'pos' : 'neg';
                const sign = m.effect >= 0 ? '+' : '';
                const tip = 'Drawing the ' + ord + ' copy changes total mana spent by '
                    + sign + fmt(m.effect, 2) + ' on average (90% CI ±' + fmt(m.ci, 2)
                    + ', based on ' + m.n_curr + ' games).';
                return '<span class="marginal ' + cls + '" data-tip="' + escapeHtml(tip) + '">'
                    + ord + ': ' + sign + fmt(m.effect, 2) + '</span>';
            });
            return '<div class="marginals">' + parts.join('') + '</div>';
        }

        function renderBadge(card) {
            const sat = card.saturation || {};
            if (sat.badge === 'scaling') {
                return '<span class="sat-badge sat-scaling" data-tip="Every extra copy you draw is still adding measurable mana. More copies would likely help.">↑ Add more</span>';
            }
            if (sat.badge === 'saturated') {
                const tip = 'Past copy ' + sat.saturates_at + ", additional copies don't add measurable value. "
                    + sat.saturates_at + ' is roughly the sweet spot.';
                return '<span class="sat-badge sat-saturated" data-tip="' + escapeHtml(tip) + '">≈ Enough at ' + sat.saturates_at + '</span>';
            }
            if (sat.badge === 'crowding') {
                return '<span class="sat-badge sat-crowding" data-tip="Drawing extra copies actually hurts (they crowd your hand or compete for mana). Consider running fewer.">↓ Cut copies</span>';
            }
            if (sat.badge === 'unclear') {
                return '<span class="sat-badge sat-unclear" data-tip="Not enough games drew this spell at varying counts to give a confident verdict. Try running more simulations.">? Need more data</span>';
            }
            return '';
        }

        function renderRow(card, i, scoreClass) {
            const label = escapeHtml(card.label || card.name);
            const copies = card.copies || 1;
            const tip = card.always_drawn
                ? 'Drawn in nearly every game, so there is no "didn\'t draw it" group to compare against. Score shown is the sum of the per-copy effects that were statistically significant.'
                : 'Average mana spent in games that drew this spell (' + fmt(card.mean_with, 2)
                    + ") minus games that didn't (" + fmt(card.mean_without, 2) + ').';
            const marker = card.always_drawn ? '<sup class="always-drawn">∑</sup>' : '';
            let row = '<tr><td>' + (i + 1) + '</td>';
            row += '<td style="text-align:left"><div>' + label + '</div>';
            row += '<small>e.g. ' + cardLink(card.name) + '</small></td>';
            row += '<td>' + copies + '</td>';
            row += '<td>' + escapeHtml(card.cost) + '</td>';
            row += '<td class="' + scoreClass + '" title="' + escapeHtml(tip) + '">' + (card.score >= 0 ? '+' : '') + fmt(card.score, 2) + marker + '</td>';
            row += '<td style="text-align:left">' + renderMarginals(card) + '</td>';
            row += '<td>' + renderBadge(card) + '</td></tr>';
            return row;
        }

        const headerCells = '<th>#</th>'
            + '<th>Spell</th>'
            + '<th title="How many copies of this spell (or simulator-equivalent variants) are in the deck.">Copies</th>'
            + '<th title="Average mana actually paid when this spell was cast across all simulated games.">Avg cost</th>'
            + '<th title="Average mana spent in games that drew at least one copy minus games that drew none. Positive = the deck performs better with this spell in hand.">Impact</th>'
            + '<th data-tip="Per-copy effect: how much more (or less) mana the deck spends when you draw the 1st, 2nd, 3rd … copy compared to having one fewer. Hover any pill for details.">Each extra copy <span class="help-icon" aria-hidden="true">?</span></th>'
            + '<th data-tip="Plain-English verdict on whether to add more copies, you have enough, cut copies, or there isn\'t enough data yet.">Recommendation <span class="help-icon" aria-hidden="true">?</span></th>';

        // High performers
        html += '<div><h3>Top Performers</h3><div class="table-wrap"><table class="stats-table">';
        html += '<thead><tr>' + headerCells + '</tr></thead><tbody>';
        cp.high_performing.forEach((card, i) => {
            html += renderRow(card, i, 'score-positive');
        });
        html += '</tbody></table></div></div>';

        // Low performers
        html += '<div><h3>Low Performers</h3><div class="table-wrap"><table class="stats-table">';
        html += '<thead><tr>' + headerCells + '</tr></thead><tbody>';
        cp.low_performing.forEach((card, i) => {
            html += renderRow(card, i, 'score-negative');
        });
        html += '</tbody></table></div></div></div>';
        return html;
    }

    function renderChartCanvases() {
        return `<h2>Charts</h2>
        <div class="charts-grid">
            <div class="chart-container"><canvas id="manaChart"></canvas></div>
            <div class="chart-container"><canvas id="consistencyChart"></canvas></div>
        </div>`;
    }

    function renderReplayHTML(results) {
        if (!results[0].replay_data || !results[0].replay_data.top
            || results[0].replay_data.top.length === 0) return '';

        return `<h2>Game Replays</h2>
        <div class="replay-container" id="replay-viewer">
            <div class="replay-tabs" id="replay-tabs">
                <button class="replay-tab active" data-quantile="top">Top Quartile</button>
                <button class="replay-tab" data-quantile="mid">Mid</button>
                <button class="replay-tab" data-quantile="low">Low Quartile</button>
            </div>
            <div class="replay-games" id="replay-games"></div>
            <div class="replay-info" id="replay-info"></div>
            <div class="replay-nav" id="replay-nav">
                <button id="replay-prev">&lt; Prev</button>
                <span class="turn-counter" id="replay-turn-counter"></span>
                <button id="replay-next">Next &gt;</button>
            </div>
            <div id="replay-content">
                <div class="replay-section">
                    <h4>Hand (before draw):</h4>
                    <div class="replay-card-list" id="replay-hand-before"></div>
                </div>
                <div class="replay-section">
                    <h4>Played this turn:</h4>
                    <div class="replay-card-list" id="replay-played"></div>
                </div>
                <div class="replay-section">
                    <h4>Board State:</h4>
                    <div class="replay-card-list" id="replay-board"></div>
                </div>
            </div>
        </div>`;
    }

    // -- Chart rendering --

    function renderCharts(data) {
        const labels = data.map(d => d.land_count);

        // Destroy existing charts
        ['manaChart', 'consistencyChart'].forEach(id => {
            const existing = Chart.getChart(id);
            if (existing) existing.destroy();
        });

        // Mana EV
        new Chart(document.getElementById('manaChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {label: 'Mean Mana', data: data.map(d => d.mean_mana),
                     borderColor: '#2563eb', backgroundColor: '#2563eb', borderWidth: 2, fill: false},
                    {label: '75th Percentile', data: data.map(d => d.percentile_75),
                     borderColor: 'rgba(37, 99, 235, 0.3)', backgroundColor: 'rgba(37, 99, 235, 0.1)',
                     borderWidth: 1, fill: '+1'},
                    {label: '50th Percentile', data: data.map(d => d.percentile_50),
                     borderColor: 'rgba(37, 99, 235, 0.5)', backgroundColor: 'rgba(37, 99, 235, 0.1)',
                     borderWidth: 1, fill: false, borderDash: [5, 5]},
                    {label: '25th Percentile', data: data.map(d => d.percentile_25),
                     borderColor: 'rgba(37, 99, 235, 0.3)', backgroundColor: 'rgba(37, 99, 235, 0.1)',
                     borderWidth: 1, fill: '-1'},
                ]
            },
            options: {
                responsive: true,
                plugins: {title: {display: true, text: 'Mana EV by Land Count'}},
                scales: {
                    x: {title: {display: true, text: 'Land Count'}},
                    y: {title: {display: true, text: 'Total Mana Spent'}}
                }
            }
        });

        // Consistency
        new Chart(document.getElementById('consistencyChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Consistency',
                    data: data.map(d => d.consistency),
                    borderColor: '#16a34a', backgroundColor: '#16a34a',
                    borderWidth: 2, fill: false,
                }]
            },
            options: {
                responsive: true,
                plugins: {title: {display: true, text: 'Consistency Score by Land Count'}},
                scales: {
                    x: {title: {display: true, text: 'Land Count'}},
                    y: {title: {display: true, text: 'Consistency'}, min: 0, max: 1.2}
                }
            }
        });
    }

    // -- Replay viewer --

    function initReplayViewer(data) {
        const viewer = document.getElementById('replay-viewer');
        if (!viewer) return;

        const replayData = data[0].replay_data;
        if (!replayData || !replayData.top || replayData.top.length === 0) return;

        let currentQuantile = 'top';
        let currentGame = 0;
        let currentTurn = 0;

        function renderGameButtons() {
            const games = replayData[currentQuantile] || [];
            const container = document.getElementById('replay-games');
            container.innerHTML = '';
            for (let i = 0; i < games.length; i++) {
                const btn = document.createElement('button');
                btn.className = 'replay-game-btn' + (i === currentGame ? ' active' : '');
                btn.textContent = i + 1;
                btn.addEventListener('click', function() {
                    currentGame = i;
                    currentTurn = 0;
                    renderReplay();
                });
                container.appendChild(btn);
            }
        }

        function renderReplay() {
            const games = replayData[currentQuantile] || [];
            if (games.length === 0) {
                document.getElementById('replay-info').textContent = 'No games in this bucket.';
                document.getElementById('replay-turn-counter').textContent = '';
                document.getElementById('replay-hand-before').innerHTML = '';
                document.getElementById('replay-played').innerHTML = '';
                document.getElementById('replay-board').innerHTML = '';
                renderGameButtons();
                return;
            }

            const game = games[currentGame];
            const turn = game.turns[currentTurn];

            document.querySelectorAll('.replay-tab').forEach(tab => {
                tab.classList.toggle('active', tab.dataset.quantile === currentQuantile);
            });

            renderGameButtons();

            document.getElementById('replay-info').innerHTML =
                '<strong>Mana:</strong> ' + game.total_mana
                + ' &nbsp;|&nbsp; <strong>Mulligans:</strong> ' + game.mulligans
                + ' &nbsp;|&nbsp; <strong>Starting hand:</strong> '
                + game.starting_hand.map(cardLink).join(', ');

            document.getElementById('replay-turn-counter').textContent =
                'Turn ' + turn.turn + ' of ' + game.turns.length;
            document.getElementById('replay-prev').disabled = currentTurn === 0;
            document.getElementById('replay-next').disabled = currentTurn === game.turns.length - 1;

            document.getElementById('replay-hand-before').innerHTML =
                turn.hand_before_draw.length > 0
                    ? turn.hand_before_draw.map(cardLink).join(', ')
                    : '<em>Empty</em>';

            const playedHtml = turn.played.map(function(c) {
                const cls = 'replay-played-card' + (c.is_land ? ' is-land' : '');
                const detail = c.is_land ? '(land)' : '(' + escapeHtml(c.cost) + ', ' + c.mana_spent + ' mana)';
                return '<span class="' + cls + '">' + cardLink(c.name) + ' ' + detail + '</span>';
            }).join(' ');
            document.getElementById('replay-played').innerHTML = playedHtml || '<em>Nothing played</em>';

            const boardParts = [];
            boardParts.push('<strong>Mana spent:</strong> ' + turn.mana_spent_this_turn
                + ' &nbsp;|&nbsp; <strong>Total production:</strong> ' + turn.total_mana_production);
            boardParts.push('<br><strong>Battlefield:</strong> '
                + (turn.battlefield.length > 0 ? turn.battlefield.map(cardLink).join(', ') : '<em>Empty</em>'));
            boardParts.push('<br><strong>Lands:</strong> '
                + (turn.lands.length > 0 ? turn.lands.map(cardLink).join(', ') : '<em>None</em>'));
            boardParts.push('<br><strong>Hand:</strong> '
                + (turn.hand_after.length > 0 ? turn.hand_after.map(cardLink).join(', ') : '<em>Empty</em>'));
            if (turn.graveyard.length > 0) {
                boardParts.push('<br><strong>Graveyard:</strong> ' + turn.graveyard.map(cardLink).join(', '));
            }
            document.getElementById('replay-board').innerHTML = boardParts.join('');

            rebindTooltips();
        }

        document.querySelectorAll('.replay-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                currentQuantile = this.dataset.quantile;
                currentGame = 0;
                currentTurn = 0;
                renderReplay();
            });
        });

        document.getElementById('replay-prev').addEventListener('click', function() {
            if (currentTurn > 0) { currentTurn--; renderReplay(); }
        });
        document.getElementById('replay-next').addEventListener('click', function() {
            const games = replayData[currentQuantile] || [];
            if (games.length > 0 && currentTurn < games[currentGame].turns.length - 1) {
                currentTurn++;
                renderReplay();
            }
        });

        renderReplay();
    }

    // -- Feature Analysis --

    /**
     * Render recommendation text, inserting a Scryfall hover link for the
     * example card when present (reuses the existing card-link tooltip system).
     */
    function renderRecText(r) {
        if (!r.example_card) {
            return escapeHtml(r.recommendation);
        }
        var name = r.example_card.name;
        var text = r.recommendation;
        var idx = text.indexOf(name);
        if (idx === -1) {
            return escapeHtml(text);
        }
        var before = text.substring(0, idx);
        var after = text.substring(idx + name.length);
        return escapeHtml(before) + cardLink(name) + escapeHtml(after);
    }

    function renderFeatureAnalysis(results) {
        const analysis = results[0] && results[0].feature_analysis;
        if (!analysis || !analysis.recommendations || analysis.recommendations.length === 0) {
            return '';
        }

        let html = '<div class="feature-analysis-section">';
        html += '<h2>Recommended Changes</h2>';
        html += '<p class="hint">Based on analysis of ' + analysis.n_configs
            + ' configurations evaluated during optimization.</p>';

        // Synthesized recommendations
        const recs = analysis.recommendations;
        const positiveRecs = recs.filter(function(r) { return r.impact > 0; });
        const negativeRecs = recs.filter(function(r) { return r.impact < 0; });

        if (positiveRecs.length > 0) {
            html += '<div class="recommendations-list">';
            html += '<h3>Changes that improve results</h3>';
            html += '<ul class="rec-list">';
            for (let i = 0; i < Math.min(positiveRecs.length, 8); i++) {
                const r = positiveRecs[i];
                const badge = r.confidence === 'high' ? 'rec-badge-high'
                    : r.confidence === 'medium' ? 'rec-badge-med' : 'rec-badge-low';
                html += '<li class="rec-item rec-positive">';
                html += '<span class="rec-badge ' + badge + '">' + r.confidence + '</span> ';
                html += '<strong>' + escapeHtml(r.label) + '</strong>: ';
                html += renderRecText(r);
                html += ' <span class="rec-delta">(';
                html += r.impact > 0 ? '+' : '';
                html += fmt(r.impact, 4) + ')</span>';
                html += '</li>';
            }
            html += '</ul></div>';
        }

        if (negativeRecs.length > 0) {
            html += '<div class="recommendations-list">';
            html += '<h3>Changes that hurt results</h3>';
            html += '<ul class="rec-list">';
            for (let i = 0; i < Math.min(negativeRecs.length, 5); i++) {
                const r = negativeRecs[i];
                const badge = r.confidence === 'high' ? 'rec-badge-high'
                    : r.confidence === 'medium' ? 'rec-badge-med' : 'rec-badge-low';
                html += '<li class="rec-item rec-negative">';
                html += '<span class="rec-badge ' + badge + '">' + r.confidence + '</span> ';
                html += '<strong>' + escapeHtml(r.label) + '</strong>: ';
                html += renderRecText(r);
                html += ' <span class="rec-delta">(';
                html += r.impact > 0 ? '+' : '';
                html += fmt(r.impact, 4) + ')</span>';
                html += '</li>';
            }
            html += '</ul></div>';
        }

        // Advanced statistics dropdown
        html += '<details class="advanced-stats">';
        html += '<summary>Advanced Statistics</summary>';

        // Regression summary
        const reg = analysis.regression;
        if (reg) {
            html += '<div class="stats-subsection">';
            html += '<h4>Regression Analysis '
                + (reg.weighted ? '(Weighted Least Squares)' : '(OLS)')
                + '</h4>';
            html += '<p>R&sup2; = ' + fmt(reg.r_squared, 4)
                + ' &mdash; model explains ' + fmt(reg.r_squared * 100, 1)
                + '% of score variance.</p>';
            html += '<div class="table-wrap"><table class="stats-table"><thead><tr>';
            html += '<th>Feature</th><th>Coefficient</th><th>Std Beta</th>';
            html += '</tr></thead><tbody>';
            for (let i = 0; i < reg.coefficients.length; i++) {
                const c = reg.coefficients[i];
                const cls = c.coefficient > 0 ? 'score-positive' : c.coefficient < 0 ? 'score-negative' : '';
                html += '<tr>';
                html += '<td style="text-align:left">' + escapeHtml(c.label || c.feature) + '</td>';
                html += '<td class="' + cls + '">' + (c.coefficient >= 0 ? '+' : '') + fmt(c.coefficient, 4) + '</td>';
                html += '<td>' + fmt(c.std_beta, 4) + '</td>';
                html += '</tr>';
            }
            html += '</tbody></table></div></div>';
        }

        // Marginal impact table
        const marginal = analysis.marginal_impact;
        if (marginal && marginal.length > 0) {
            html += '<div class="stats-subsection">';
            html += '<h4>Marginal Feature Impact</h4>';
            html += '<div class="table-wrap"><table class="stats-table"><thead><tr>';
            html += '<th>Feature</th><th>Delta</th><th>Mean With</th><th>Mean Without</th><th>Count</th>';
            html += '</tr></thead><tbody>';
            for (let i = 0; i < marginal.length; i++) {
                const m = marginal[i];
                const cls = m.delta > 0 ? 'score-positive' : m.delta < 0 ? 'score-negative' : '';
                html += '<tr>';
                html += '<td style="text-align:left">' + escapeHtml(m.label) + '</td>';
                html += '<td class="' + cls + '">' + (m.delta >= 0 ? '+' : '') + fmt(m.delta, 4) + '</td>';
                html += '<td>' + fmt(m.mean_with, 4) + '</td>';
                html += '<td>' + fmt(m.mean_without, 4) + '</td>';
                html += '<td>' + m.count + '</td>';
                html += '</tr>';
            }
            html += '</tbody></table></div></div>';
        }

        html += '</details>';
        html += '</div>';
        return html;
    }

    // -- Mana view toggle --

    function initManaViewToggle(container) {
        container.querySelectorAll('.mana-view-toggle').forEach(function(toggle) {
            toggle.querySelectorAll('.mana-view-btn').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var view = this.dataset.view;
                    toggle.dataset.current = view;
                    toggle.querySelectorAll('.mana-view-btn').forEach(function(b) {
                        b.classList.toggle('active', b.dataset.view === view);
                    });
                    var table = toggle.closest('table');
                    var dataKey = view === 'bottom_25' ? 'bot' : view === 'top_25' ? 'top' : 'mean';
                    table.querySelectorAll('.mana-cell').forEach(function(cell) {
                        var val = cell.dataset[dataKey];
                        var ci = cell.dataset.ci;
                        cell.innerHTML = val + ' <small>&plusmn;' + ci + '</small>';
                    });
                });
            });
        });
    }

    // -- Public API --

    /**
     * Render simulation results into a container element.
     *
     * @param {HTMLElement} container - Target element to render into
     * @param {Array} results - Array of result dicts (from result_to_dict)
     * @param {string} deckName - Deck name for the title
     */
    function render(container, results, deckName) {
        const isOptimization = results.length > 0 && results[0].opt_config !== undefined;

        let html = '<div class="results-content">';
        html += '<h1>Results: ' + escapeHtml(deckName) + '</h1>';

        html += renderDeckScore(results);
        if (isOptimization) {
            html += renderFeatureAnalysis(results);
        }
        html += renderSummaryTable(results, isOptimization);
        html += renderCardPerformance(results);
        if (!isOptimization) {
            html += renderChartCanvases();
        }
        html += renderReplayHTML(results);
        html += '</div>';

        container.innerHTML = html;

        // Render interactive components after DOM is updated
        renderDeckScoreChart(results);
        if (!isOptimization) {
            renderCharts(results);
        }
        initManaViewToggle(container);
        initReplayViewer(results);
        rebindTooltips();
    }

    return {render, rebindTooltips, renderCharts, initReplayViewer};
})();
