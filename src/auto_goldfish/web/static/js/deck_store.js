/**
 * DeckStore — localStorage CRUD for user deck data.
 *
 * Each deck is stored as {cards: [...], overrides: {...}, last_accessed: <ms>}.
 * All decks live under a single localStorage key.
 * When storage is full, the least-recently-accessed deck is evicted.
 */
var DeckStore = {
    _key: 'ag_decks',

    _load: function() {
        try {
            return JSON.parse(localStorage.getItem(this._key)) || {};
        } catch (e) {
            return {};
        }
    },

    _save: function(data) {
        try {
            localStorage.setItem(this._key, JSON.stringify(data));
        } catch (e) {
            if (e.name === 'QuotaExceededError' || e.code === 22) {
                if (this._evictOldest(data)) {
                    this._save(data);
                } else {
                    alert('localStorage is full and no decks could be evicted.');
                }
            }
        }
    },

    /** Remove the least-recently-accessed deck from data (mutates). Returns true if one was removed. */
    _evictOldest: function(data) {
        var oldest = null;
        var oldestTime = Infinity;
        for (var name in data) {
            if (!data.hasOwnProperty(name)) continue;
            var t = data[name].last_accessed || 0;
            if (t < oldestTime) {
                oldestTime = t;
                oldest = name;
            }
        }
        if (oldest) {
            delete data[oldest];
            return true;
        }
        return false;
    },

    listDecks: function() {
        var data = this._load();
        var result = [];
        for (var name in data) {
            if (!data.hasOwnProperty(name)) continue;
            var deck = data[name];
            var cards = deck.cards || [];
            var commanders = [];
            var landCount = 0;
            for (var i = 0; i < cards.length; i++) {
                if (cards[i].commander) commanders.push(cards[i].name);
                if (cards[i].types && cards[i].types.indexOf('Land') !== -1) {
                    landCount += cards[i].quantity || 1;
                }
            }
            result.push({
                name: name,
                card_count: cards.length,
                commanders: commanders,
                land_count: landCount
            });
        }
        return result;
    },

    getDeck: function(name) {
        var data = this._load();
        if (!data[name]) return null;
        data[name].last_accessed = Date.now();
        this._save(data);
        return data[name];
    },

    saveDeck: function(name, cards, overrides, deckUrl) {
        var data = this._load();
        var entry = {cards: cards, overrides: overrides || {}, last_accessed: Date.now()};
        if (deckUrl) entry.deck_url = deckUrl;
        data[name] = entry;
        this._save(data);
    },

    saveOverrides: function(name, overrides) {
        var data = this._load();
        if (data[name]) {
            data[name].overrides = overrides;
            data[name].last_accessed = Date.now();
            this._save(data);
        }
    },

    deleteDeck: function(name) {
        var data = this._load();
        delete data[name];
        this._save(data);
    },

    hasDeck: function(name) {
        var data = this._load();
        return name in data;
    }
};

/**
 * Leaderboard — localStorage-based top-10 tracking across three metrics.
 */
var Leaderboard = {
    _key: 'ag_leaderboard',

    _empty: function() {
        return {mana_spent: [], consistency: [], cards_drawn: []};
    },

    _load: function() {
        try {
            var data = JSON.parse(localStorage.getItem(this._key));
            if (data && data.mana_spent) return data;
            return this._empty();
        } catch (e) {
            return this._empty();
        }
    },

    _save: function(data) {
        try {
            localStorage.setItem(this._key, JSON.stringify(data));
        } catch (e) {}
    },

    /**
     * Update leaderboard after a simulation run.
     * @param {string} deckName
     * @param {string} deckUrl - Archidekt URL
     * @param {Array} results - array of result entries (one per land count)
     * @param {Object} effectOverrides - merged overrides dict
     */
    update: function(deckName, deckUrl, results, effectOverrides) {
        if (!results || !results.length) return;

        // Count ramp and draw spells from overrides
        var rampCount = 0;
        var drawCount = 0;
        if (effectOverrides) {
            for (var cardName in effectOverrides) {
                if (!effectOverrides.hasOwnProperty(cardName)) continue;
                var entry = effectOverrides[cardName];
                var effects = Array.isArray(entry) ? entry : (entry && Array.isArray(entry.categories) ? entry.categories : null);
                if (!effects) continue;
                var hasRamp = false, hasDraw = false;
                for (var i = 0; i < effects.length; i++) {
                    if (effects[i].category === 'ramp') hasRamp = true;
                    if (effects[i].category === 'draw') hasDraw = true;
                }
                if (hasRamp) rampCount++;
                if (hasDraw) drawCount++;
            }
        }

        // Find best entry for each metric
        var bestMana = results[0], bestCons = results[0], bestDraw = results[0];
        for (var j = 1; j < results.length; j++) {
            if (results[j].mean_mana > bestMana.mean_mana) bestMana = results[j];
            if (results[j].consistency > bestCons.consistency) bestCons = results[j];
            if (results[j].mean_draws > bestDraw.mean_draws) bestDraw = results[j];
        }

        var board = this._load();
        var metrics = [
            {key: 'mana_spent', best: bestMana, field: 'mean_mana'},
            {key: 'consistency', best: bestCons, field: 'consistency'},
            {key: 'cards_drawn', best: bestDraw, field: 'mean_draws'},
        ];

        for (var m = 0; m < metrics.length; m++) {
            var cat = metrics[m].key;
            var value = metrics[m].best[metrics[m].field];
            var landCount = metrics[m].best.land_count;
            var list = board[cat];

            // Upsert by deck_url + deck_name
            var found = false;
            for (var k = 0; k < list.length; k++) {
                if (list[k].deck_url === deckUrl && list[k].deck_name === deckName) {
                    if (value > list[k].value) {
                        list[k].value = value;
                        list[k].land_count = landCount;
                        list[k].ramp_count = rampCount;
                        list[k].draw_count = drawCount;
                    }
                    found = true;
                    break;
                }
            }
            if (!found) {
                list.push({
                    deck_url: deckUrl,
                    deck_name: deckName,
                    value: value,
                    land_count: landCount,
                    ramp_count: rampCount,
                    draw_count: drawCount,
                });
            }

            // Sort descending by value, trim to 10
            list.sort(function(a, b) { return b.value - a.value; });
            board[cat] = list.slice(0, 10);
        }

        this._save(board);
    },

    getAll: function() {
        return this._load();
    },

    clear: function() {
        this._save(this._empty());
    }
};

/**
 * Navigate to the simulation page for a local deck.
 * POSTs deck data as JSON, replaces the page with the response.
 */
async function navigateToSim(deckName) {
    var deck = DeckStore.getDeck(deckName);
    if (!deck) { alert('Deck not found in local storage'); return; }
    var resp = await fetch('/sim/' + encodeURIComponent(deckName), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cards: deck.cards, overrides: deck.overrides})
    });
    if (!resp.ok) { alert('Failed to load simulation page'); return; }
    var html = await resp.text();
    document.open(); document.write(html); document.close();
    history.pushState(null, '', '/sim/' + encodeURIComponent(deckName));
}

/**
 * Navigate to the mana model page for a local deck.
 * POSTs deck data as JSON, replaces the page with the response.
 */
async function navigateToManaModel(deckName) {
    var deck = DeckStore.getDeck(deckName);
    if (!deck) { alert('Deck not found in local storage'); return; }
    var resp = await fetch('/mana-model/' + encodeURIComponent(deckName), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cards: deck.cards, overrides: deck.overrides})
    });
    if (!resp.ok) { alert('Failed to load mana model page'); return; }
    var html = await resp.text();
    document.open(); document.write(html); document.close();
    history.pushState(null, '', '/mana-model/' + encodeURIComponent(deckName));
}

/**
 * Navigate to the deck view page for a local deck.
 * POSTs deck data as JSON, replaces the page with the response.
 */
async function navigateToDeckView(deckName) {
    var deck = DeckStore.getDeck(deckName);
    if (!deck) { alert('Deck not found in local storage'); return; }
    var resp = await fetch('/decks/' + encodeURIComponent(deckName), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cards: deck.cards})
    });
    if (!resp.ok) { alert('Failed to load deck view'); return; }
    var html = await resp.text();
    document.open(); document.write(html); document.close();
    history.pushState(null, '', '/decks/' + encodeURIComponent(deckName));
}
