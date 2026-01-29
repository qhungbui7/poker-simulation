import itertools
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

RANKS = list(range(2, 15))
SUITS = "cdhs"
HUMAN_ID = 0

MODE = "learn"
MC_SAMPLES = 1400
BET_CANDIDATES = (0.5, 0.75, 1.0)


def format_card(card):
    r, s = card
    m = {11: "J", 12: "Q", 13: "K", 14: "A"}
    sm = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
    return f"{m.get(r, r)}{sm.get(s, s)}"


def format_cards(cards):
    return " ".join(format_card(c) for c in cards)


def make_deck():
    return [(r, s) for r in RANKS for s in SUITS]


def deal(deck, n):
    return [deck.pop() for _ in range(n)]


def best_hand_7(cards):
    def eval5(c):
        ranks = [r for r, _ in c]
        rc = Counter(ranks)
        sc = Counter(s for _, s in c)
        uniq_desc = sorted(set(ranks), reverse=True)

        def straight_high(desc):
            for i in range(len(desc) - 4):
                if desc[i] - desc[i + 4] == 4:
                    return desc[i]
            if desc and desc[0] == 14:
                has = 0
                for r in desc:
                    if r == 5:
                        has |= 1
                    elif r == 4:
                        has |= 2
                    elif r == 3:
                        has |= 4
                    elif r == 2:
                        has |= 8
                if has == 15:
                    return 5
            return None

        flush_suit = next((s for s, cnt in sc.items() if cnt >= 5), None)

        if flush_suit:
            fr = sorted(set(r for r, s in c if s == flush_suit), reverse=True)
            sh = straight_high(fr)
            if sh is not None:
                return (8, (sh,))

        sh = straight_high(uniq_desc)
        counts = sorted(rc.items(), key=lambda x: (x[1], x[0]), reverse=True)

        if counts[0][1] == 4:
            quad = counts[0][0]
            kicker = max(r for r in uniq_desc if r != quad)
            return (7, (quad, kicker))

        if counts[0][1] == 3 and len(counts) > 1 and counts[1][1] >= 2:
            return (6, (counts[0][0], counts[1][0]))

        if flush_suit:
            top5 = sorted([r for r, s in c if s == flush_suit], reverse=True)[:5]
            return (5, tuple(top5))

        if sh is not None:
            return (4, (sh,))

        if counts[0][1] == 3:
            trips = counts[0][0]
            kickers = tuple([r for r in uniq_desc if r != trips][:2])
            return (3, (trips,) + kickers)

        if len(counts) >= 2 and counts[0][1] == 2 and counts[1][1] == 2:
            hi, lo = counts[0][0], counts[1][0]
            kicker = next(r for r in uniq_desc if r not in (hi, lo))
            return (2, (hi, lo, kicker))

        if counts[0][1] == 2:
            pair = counts[0][0]
            kickers = tuple([r for r in uniq_desc if r != pair][:3])
            return (1, (pair,) + kickers)

        return (0, tuple(uniq_desc[:5]))

    best = (-1, ())
    for c in itertools.combinations(cards, 5):
        v = eval5(c)
        if v > best:
            best = v
    return best


# preflop scoring + combos

def hole_strength_preflop(hole):
    (r1, s1), (r2, s2) = hole
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    pair = hi == lo
    suited = s1 == s2
    gap = hi - lo
    score = 0.0
    if pair:
        score += 5.0 + (hi - 2) * 0.55
    else:
        score += (hi - 2) * 0.35 + (lo - 2) * 0.15
        if suited:
            score += 0.9
        if gap == 1:
            score += 0.55
        elif gap == 2:
            score += 0.25
        elif gap >= 5:
            score -= 0.4
        if hi >= 13 and lo >= 10:
            score += 0.4
    return score


ALL_COMBOS = []
ALL_SCORES = []


def all_hole_combos():
    deck = make_deck()
    combos = []
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            h = (deck[i], deck[j])
            combos.append((hole_strength_preflop(h), h))
    combos.sort(key=lambda x: x[0])
    return combos


ALL_COMBOS = all_hole_combos()
ALL_SCORES = [s for s, _ in ALL_COMBOS]


def range_threshold(percent):
    if percent <= 0:
        return 10**9
    if percent >= 1:
        return -10**9
    cut = int((1.0 - percent) * len(ALL_SCORES))
    if cut < 0:
        cut = 0
    if cut >= len(ALL_SCORES):
        cut = len(ALL_SCORES) - 1
    return ALL_SCORES[cut]


# board helpers

def board_texture(board):
    if not board:
        return {"paired": False, "two_tone": False, "monotone": False, "connected": 0, "high": None, "summary": "preflop", "favours": "unknown"}
    ranks = sorted({r for r, _ in board})
    suits = [s for _, s in board]
    sc = Counter(suits)
    paired = len(ranks) < len(board)
    monotone = max(sc.values()) >= 3
    two_tone = (not monotone) and max(sc.values()) == 2
    ranks_desc = sorted(ranks, reverse=True)
    connected = 0
    for i in range(len(ranks_desc) - 1):
        d = ranks_desc[i] - ranks_desc[i + 1]
        if d == 1:
            connected += 2
        elif d == 2:
            connected += 1
    hi = max(ranks) if ranks else None
    if paired:
        summary = "paired"
    elif monotone:
        summary = "monotone"
    elif two_tone:
        summary = "two-tone"
    elif connected >= 3:
        summary = "connected"
    elif hi is not None and hi >= 13:
        summary = "high-card"
    else:
        summary = "dry"
    favours = "unknown"
    if summary in ("high-card", "dry"):
        favours = "preflop aggressor"
    if summary in ("connected",):
        favours = "caller / wide ranges"
    if summary in ("monotone", "two-tone"):
        favours = "range-dependent"
    if summary in ("paired",):
        favours = "preflop aggressor (slightly)"
    return {"paired": paired, "two_tone": two_tone, "monotone": monotone, "connected": connected, "high": hi, "summary": summary, "favours": favours}


def flush_draw_info(hole, board):
    suits = [s for _, s in board] + [s for _, s in hole]
    sc = Counter(suits)
    best_suit, cnt = sc.most_common(1)[0]
    return best_suit, cnt


def straight_draw_flags(hole, board):
    ranks = sorted(set([r for r, _ in board] + [r for r, _ in hole]))
    if not ranks:
        return False, False

    def has_run_of_len(L):
        for i in range(len(ranks) - (L - 1)):
            if ranks[i + (L - 1)] - ranks[i] == (L - 1):
                return True
        return False

    open_ended = has_run_of_len(4)
    gutshot = False
    rs = set(ranks)
    for start in range(2, 11):
        window = {start, start + 1, start + 2, start + 3, start + 4}
        inter = len(window & rs)
        if inter == 4:
            gutshot = True
            break
    if 14 in rs:
        wheel_window = {14, 2, 3, 4, 5}
        if len(wheel_window & rs) == 4:
            gutshot = True
    return open_ended, gutshot


def blocker_hints(hole, board):
    if len(board) < 3:
        return {"flush_blocker": None, "note": ""}
    sc = Counter([s for _, s in board])
    suit, cnt = sc.most_common(1)[0]
    if cnt < 3:
        return {"flush_blocker": None, "note": ""}
    suited_ranks = {r for r, s in hole if s == suit}
    if 14 in suited_ranks:
        return {"flush_blocker": f"A{format_card((14, suit))[1]}", "note": "Blocks nut flush for this suit."}
    if 13 in suited_ranks:
        return {"flush_blocker": f"K{format_card((13, suit))[1]}", "note": "Blocks 2nd-nut flush for this suit."}
    if 12 in suited_ranks:
        return {"flush_blocker": f"Q{format_card((12, suit))[1]}", "note": "Blocks strong flushes for this suit."}
    return {"flush_blocker": None, "note": ""}


# human-style estimators

def count_outs(hole, board):
    outs = 0
    suit, cnt = flush_draw_info(hole, board)
    if cnt == 4:
        outs += 9
    open_ended, gutshot = straight_draw_flags(hole, board)
    if open_ended:
        outs += 8
    elif gutshot:
        outs += 4
    return outs


def classify_hero_simple(hole, board):
    v = best_hand_7(list(hole) + board)[0]
    return v


def human_estimate_equity(hero_hole, board, opp_count):
    cls = classify_hero_simple(hero_hole, board)
    base_map = {8:0.98,7:0.88,6:0.80,5:0.75,4:0.70,3:0.60,2:0.55,1:0.40,0:0.12}
    eq = base_map.get(cls, 0.2)
    outs = count_outs(hero_hole, board)
    if outs:
        add = min(outs * 0.04, 0.40)
        eq = max(eq, add)
    multiway_penalty = 0.90 if opp_count >= 2 else 1.0
    return eq * multiway_penalty


# multiprocessing MC worker

def _mc_worker(args):
    hero_hole, board, thresholds, deck_rem, samples, seed = args
    rng = random.Random(seed)
    used = set(hero_hole)
    used.update(board)
    wins = ties = 0
    need_board = 5 - len(board)
    for _ in range(samples):
        banned = set(used)
        opp_holes = []
        for t in thresholds:
            while True:
                idx = rng.randrange(len(ALL_COMBOS))
                score, hole = ALL_COMBOS[idx]
                if score < t:
                    continue
                c1, c2 = hole
                if c1 in banned or c2 in banned:
                    continue
                opp_holes.append(hole)
                banned.add(hole[0]); banned.add(hole[1])
                break
        runout = rng.sample([c for c in deck_rem if c not in banned], need_board)
        full_board = board + runout
        hero_score = best_hand_7(list(hero_hole) + full_board)
        opp_best = max(best_hand_7(list(h) + full_board) for h in opp_holes)
        if hero_score > opp_best:
            wins += 1
        elif hero_score == opp_best:
            ties += 1
    return wins, ties


# orchestrator (thresholds passed in)

def mc_estimate_equity(hero_hole, board, thresholds, samples, seed):
    if not thresholds:
        return 1.0
    deck_all = make_deck()
    used = set(hero_hole)
    used.update(board)
    deck_rem = [c for c in deck_all if c not in used]
    cpu = min(multiprocessing.cpu_count(), 8)
    workers = cpu
    chunk = max(1, samples // workers)
    args_list = []
    for i in range(workers):
        s = chunk if i < workers - 1 else samples - chunk * (workers - 1)
        args_list.append((hero_hole, board, thresholds, deck_rem, s, seed + i + 1))
    wins = ties = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for w, t in ex.map(_mc_worker, args_list):
            wins += w
            ties += t
    return (wins + 0.5 * ties) / samples


class AdaptiveBot:
    def action(self, hole, board, to_call, pot, stack, valid_actions):
        if to_call == 0 and "check" in valid_actions:
            return "check", 0
        if to_call > 0 and "call" in valid_actions:
            return "call", 0
        return "fold", 0


class Table:
    def __init__(self, n=6, stack=1000, sb=5, bb=10, seed=0, bot=None, mode=MODE, mc_samples=MC_SAMPLES, eval_mode='both'):
        self.n = n
        self.stacks = [stack] * n
        self.sb = sb; self.bb = bb; self.btn = 0
        self.base_seed = seed
        self.rng = random.Random(seed)
        self.hand_no = 0
        self.bot = bot or AdaptiveBot()
        self.total_chips = sum(self.stacks)
        self.mode = mode
        self.mc_samples = mc_samples
        self.eval_mode = eval_mode
        self.hand_log = []
        self.showdown_log = {}
        self.range_pct = [0.55] * n
        self.preflop_aggressor = None

    def check_invariants(self):
        assert all(s >= 0 for s in self.stacks)
        assert sum(self.stacks) == self.total_chips

    def blind_positions(self):
        if self.n == 2:
            return self.btn, (self.btn + 1) % self.n
        return (self.btn + 1) % self.n, (self.btn + 2) % self.n

    def first_to_act(self, is_preflop):
        sb_pos, bb_pos = self.blind_positions()
        if self.n == 2:
            return sb_pos if is_preflop else bb_pos
        return (self.btn + 3) % self.n if is_preflop else sb_pos

    def in_hand_count(self, in_hand):
        return sum(1 for a in in_hand if a)

    def can_act_count(self, in_hand):
        return sum(1 for i, a in enumerate(in_hand) if a and self.stacks[i] > 0)

    def pot_total(self, pot, street_contrib):
        return pot + sum(street_contrib)

    def street_name(self, board, is_preflop):
        if is_preflop:
            return "preflop"
        if len(board) == 3:
            return "flop"
        if len(board) == 4:
            return "turn"
        return "river"

    def valid_actions(self, to_call, stack, can_raise):
        if to_call == 0:
            acts = ["check"]
            if can_raise and stack > 0:
                acts.append("raise")
            return acts
        acts = ["fold", "call"]
        if can_raise and stack > to_call:
            acts.append("raise")
        return acts

    def update_range_from_action(self, seat, street, action, to_call, raise_by):
        p = self.range_pct[seat]
        if street == "preflop":
            if action == "raise":
                p = min(p, 0.22)
                if self.preflop_aggressor is None:
                    self.preflop_aggressor = seat
            elif action == "call":
                p = min(p, 0.38 if to_call > 0 else 0.55)
            elif action == "check":
                p = min(p, 0.60)
        else:
            if action == "raise":
                p = min(p, 0.14 if raise_by >= max(1, to_call) else 0.20)
            elif action == "call":
                p = min(p, 0.26 if to_call > 0 else 0.40)
            elif action == "check":
                p = min(p, 0.55)
        self.range_pct[seat] = p

    def sample_hole_from_range(self, threshold, banned, rng):
        while True:
            idx = rng.randrange(len(ALL_COMBOS))
            score, hole = ALL_COMBOS[idx]
            if score < threshold:
                continue
            c1, c2 = hole
            if c1 in banned or c2 in banned:
                continue
            return hole

    def continue_probability(self, seat, street, pot_now, bet_size, to_call):
        p = self.range_pct[seat]
        tightness = 1.0 - p
        pot_now = max(1, pot_now)
        size_ratio = bet_size / pot_now
        base = {"preflop": 0.45, "flop": 0.40, "turn": 0.35, "river": 0.28}[street]
        cont = base + 0.55 * tightness - 0.35 * size_ratio
        if to_call > 0:
            price_ratio = to_call / (pot_now + to_call)
            cont -= 0.25 * price_ratio
        if cont < 0.02:
            cont = 0.02
        if cont > 0.98:
            cont = 0.98
        return cont

    def estimate_equity_vs_ranges(self, hero_hole, board, opp_seats, samples, rng):
        if not opp_seats:
            return 1.0
        thresholds = [range_threshold(self.range_pct[s]) for s in opp_seats]
        if self.eval_mode == 'human':
            return human_estimate_equity(hero_hole, board, len(opp_seats))
        if self.eval_mode == 'both':
            human = human_estimate_equity(hero_hole, board, len(opp_seats))
            mc = mc_estimate_equity(hero_hole, board, thresholds, samples, rng.randint(0, 10**9))
            return mc
        return mc_estimate_equity(hero_hole, board, thresholds, samples, rng.randint(0, 10**9))

    def hero_strength_percentile(self, hero_hole, board, opp_seats, rng):
        used = set(hero_hole)
        used.update(board)
        deck = [c for c in make_deck() if c not in used]
        sample_holes = []
        n = 40
        for _ in range(n):
            c1, c2 = rng.sample(deck, 2)
            sample_holes.append((c1, c2))
        thresholds = [range_threshold(self.range_pct[s]) for s in opp_seats]
        hero_eq = mc_estimate_equity(hero_hole, board, thresholds, samples=260, seed=rng.randint(0, 10**9))
        eqs = []
        for h in sample_holes:
            eqs.append(mc_estimate_equity(h, board, thresholds, samples=160, seed=rng.randint(0, 10**9)))
        weaker = sum(1 for e in eqs if e <= hero_eq)
        return hero_eq, weaker / len(eqs)

    def action_ev(self, hero_hole, board, in_hand, street_contrib, pot, idx, choice, raise_by, rng):
        pot_now = self.pot_total(pot, street_contrib)
        current_bet = max(street_contrib)
        to_call = current_bet - street_contrib[idx]
        street = self.street_name(board, is_preflop=(len(board) == 0))
        opp_seats = [i for i, a in enumerate(in_hand) if a and i != idx]
        thresholds = [range_threshold(self.range_pct[s]) for s in opp_seats]
        eq_all = self.estimate_equity_vs_ranges(hero_hole, board, opp_seats, samples=self.mc_samples, rng=rng)
        if choice == "fold":
            return 0.0, {"eq": eq_all}
        if choice == "check":
            return 0.0, {"eq": eq_all}
        if choice == "call":
            ev = eq_all * (pot_now + to_call) - to_call
            req = to_call / (pot_now + to_call)
            return ev, {"eq": eq_all, "req": req}
        if choice == "raise":
            invest = to_call + raise_by
            bet_size = raise_by if to_call == 0 else invest
            cont_probs = [self.continue_probability(s, street, pot_now, bet_size, to_call) for s in opp_seats]
            fold_all = 1.0
            for cp in cont_probs:
                fold_all *= (1.0 - cp)
            exp_callers = sum(cont_probs)
            if exp_callers < 0.01:
                exp_callers = 0.01
            old = [self.range_pct[s] for s in opp_seats]
            for s in opp_seats:
                self.range_pct[s] = min(self.range_pct[s], 0.28)
            eq_called = mc_estimate_equity(hero_hole, board, [range_threshold(self.range_pct[s]) for s in opp_seats], samples=max(500, self.mc_samples // 2), seed=rng.randint(0, 10**9))
            for s, v in zip(opp_seats, old):
                self.range_pct[s] = v
            pot_if_called = pot_now + invest + invest * exp_callers
            ev_foldwin = fold_all * pot_now
            ev_called = (1.0 - fold_all) * (eq_called * pot_if_called - invest)
            return ev_foldwin + ev_called, {"eq_all": eq_all, "eq_called": eq_called, "fold_all": fold_all, "exp_callers": exp_callers, "invest": invest, "pot_now": pot_now}
        return 0.0, {}

    def advise_action(self, hero_hole, board, in_hand, street_contrib, pot, idx, valid):
        pot_now = self.pot_total(pot, street_contrib)
        current_bet = max(street_contrib)
        to_call = current_bet - street_contrib[idx]
        street = self.street_name(board, is_preflop=(len(board) == 0))
        opp_seats = [i for i, a in enumerate(in_hand) if a and i != idx]
        rng = random.Random(self.base_seed + 1000003 * self.hand_no + 97 * idx + 13 * len(board))
        tex = board_texture(board)
        blockers = blocker_hints(hero_hole, board)
        eff = self.effective_stack(idx, in_hand)
        spr = eff / max(1, pot_now)
        human_eq = human_estimate_equity(hero_hole, board, len(opp_seats))
        thresholds = [range_threshold(self.range_pct[s]) for s in opp_seats]
        mc_eq = mc_estimate_equity(hero_hole, board, thresholds, samples=self.mc_samples, seed=rng.randint(0, 10**9))
        hero_eq, hero_pct = mc_eq, 0.0
        defend_freq = 1.0
        mdf_threshold = 0.0
        if to_call > 0:
            defend_freq = self.mdf_defend_freq(pot_now, to_call)
            mdf_threshold = 1.0 - defend_freq
        candidates = []
        ev_map = {}
        for act in ("fold", "check", "call"):
            if act in valid:
                ev, dbg = self.action_ev(hero_hole, board, in_hand, street_contrib, pot, idx, act, 0, rng)
                candidates.append((act, 0, ev, dbg))
                ev_map[(act, 0)] = ev
        if "raise" in valid:
            if to_call == 0:
                for frac in BET_CANDIDATES:
                    rb = max(self.bb, int(frac * max(1, pot_now)))
                    ev, dbg = self.action_ev(hero_hole, board, in_hand, street_contrib, pot, idx, "raise", rb, rng)
                    candidates.append(("raise", rb, ev, dbg))
                    ev_map[("raise", rb)] = ev
            else:
                min_inc = self.bb
                for frac in (0.5, 0.75, 1.0):
                    rb = max(min_inc, int(frac * (pot_now + to_call)))
                    ev, dbg = self.action_ev(hero_hole, board, in_hand, street_contrib, pot, idx, "raise", rb, rng)
                    candidates.append(("raise", rb, ev, dbg))
                    ev_map[("raise", rb)] = ev
        best = max(candidates, key=lambda x: x[2])
        best_action = (best[0], best[1])
        best_ev = best[2]
        if to_call > 0 and ("call" in valid) and best_action == ("fold", 0):
            call_ev = ev_map.get(("call", 0), -10**9)
            if human_eq >= mdf_threshold and call_ev > -0.03 * pot_now:
                best_action = ("call", 0)
                best_ev = call_ev
        draw_suit, suit_cnt = flush_draw_info(hero_hole, board)
        open_ended, gutshot = straight_draw_flags(hero_hole, board)
        implied = ""
        if len(board) >= 3:
            if suit_cnt == 4 or open_ended or gutshot:
                if spr >= 6:
                    implied = "Implied odds: deep SPR boosts draws."
                else:
                    implied = "Draw: implied odds limited at low SPR."
            else:
                if spr >= 8:
                    implied = "Reverse implied odds risk: deep SPR punishes thin one-pair hands."
        range_note = "unknown"
        if self.preflop_aggressor is not None and len(board) >= 3:
            if tex["favours"] == "preflop aggressor":
                range_note = f"Board favors aggressor (seat {self.preflop_aggressor})."
            elif tex["favours"] == "caller / wide ranges":
                range_note = "Board favors wider ranges (callers/connectors)."
            else:
                range_note = "Board is range-dependent."
        return {"street": street, "pot_now": pot_now, "to_call": to_call, "hero_eq_mc": mc_eq, "hero_eq_human": human_eq, "defend_freq": defend_freq, "mdf_threshold": mdf_threshold, "best_action": best_action, "best_ev": best_ev, "candidates": candidates, "texture": tex, "range_note": range_note, "blockers": blockers, "spr": spr, "implied": implied}

    def print_state(self, hands, board, in_hand, street_contrib, pot, idx):
        current_bet = max(street_contrib)
        to_call = current_bet - street_contrib[idx]
        pot_now = self.pot_total(pot, street_contrib)
        print("" + "=" * 78)
        print(f"Seat {idx} (YOU) to act")
        print(f"Board: {format_cards(board) if board else '(preflop)'}")
        print(f"Your hand: {format_cards(hands[idx])}")
        print(f"Pot: {pot_now} | To call: {to_call} | Stack: {self.stacks[idx]}")
        print("Stacks:", self.stacks)
        print("In hand:", in_hand)
        print("Street contrib:", street_contrib)

    def print_advice(self, advice):
        street = advice["street"]
        pot_now = advice["pot_now"]
        to_call = advice["to_call"]
        a, amt = advice["best_action"]
        tex = advice["texture"]
        blockers = advice["blockers"]
        print("-" * 78)
        print("Learning mode coach (human vs MC equities, ranges + fold equity + MDF + texture + blockers + SPR)")
        print(f"Street: {street} | Pot: {pot_now} | To call: {to_call} | SPR: {advice['spr']:.2f}")
        print(f"Board texture: {tex['summary']} | Favours: {tex['favours']} | {advice['range_note']}")
        if blockers["flush_blocker"] is not None:
            print(f"Blocker: {blockers['flush_blocker']} | {blockers['note']}")
        if advice["implied"]:
            print(f"{advice['implied']}")
        print(f"Hero equity (MC): {advice['hero_eq_mc']:.3f} | Hero equity (human): {advice['hero_eq_human']:.3f}")
        if to_call > 0:
            defend = advice["defend_freq"]
            foldfreq = 1.0 - defend
            print(f"MDF defend freq ≈ {defend:.2f} (fold ≈ {foldfreq:.2f}) | MDF cutoff percentile ≈ {advice['mdf_threshold']:.2f}")
        rec = a if a != "raise" else f"raise by {amt}"
        print(f"Recommended: {rec} | EV ≈ {advice['best_ev']:.2f}")
        print("Top candidates:")
        bests = sorted(advice["candidates"], key=lambda x: x[2], reverse=True)[:5]
        for act, am, ev, _dbg in bests:
            label = act if act != "raise" else f"raise {am}"
            print(f"  {label:<12} EV ≈ {ev:.2f}")
        print("-" * 78)

    def log_event(self, street, idx, action, amt, board, pot_before, to_call, advice):
        self.hand_log.append({"street": street, "seat": idx, "action": action, "amt": amt, "board": board[:], "pot_before": pot_before, "to_call": to_call, "advice": advice})

    def betting_round(self, hands, board, in_hand, street_contrib, pot, is_preflop):
        street = self.street_name(board, is_preflop)
        current_bet = max(street_contrib)
        last_full_raise = self.bb
        acted = [False] * self.n
        raise_allowed = [True] * self.n
        idx = self.first_to_act(is_preflop)
        while True:
            if self.in_hand_count(in_hand) <= 1:
                return
            if self.can_act_count(in_hand) <= 1:
                return
            done = True
            for p in range(self.n):
                if not in_hand[p] or self.stacks[p] == 0:
                    continue
                if not acted[p] or street_contrib[p] != current_bet:
                    done = False
                    break
            if done:
                return
            if not in_hand[idx] or self.stacks[idx] == 0:
                idx = (idx + 1) % self.n
                continue
            to_call = current_bet - street_contrib[idx]
            valid = self.valid_actions(to_call, self.stacks[idx], raise_allowed[idx])
            advice = None
            if idx == HUMAN_ID:
                self.print_state(hands, board, in_hand, street_contrib, pot, idx)
                if self.mode == "learn":
                    advice = self.advise_action(hands[idx], board, in_hand, street_contrib, pot, idx, valid)
                    self.print_advice(advice)
                mapping = {"r": "raise", "c": "call", "f": "fold", "k": "check"}
                while True:
                    raw = input(f"Action {valid} (r/c/f/k): ").strip().lower()
                    choice = mapping.get(raw, raw)
                    if choice in valid:
                        break
                    print(f"Invalid action '{raw}'. Valid options: {valid}")
                amt = 0
                if choice == "raise":
                    while True:
                        s = input("Raise by (chips on top of call): ").strip()
                        if s.isdigit() and int(s) > 0:
                            amt = int(s)
                            break
                        print("Enter a positive integer.")
            else:
                choice, amt = self.bot.action(hands[idx], board, to_call, pot + sum(street_contrib), self.stacks[idx], valid)
            pot_before = self.pot_total(pot, street_contrib)
            self.log_event(street, idx, choice, amt, board, pot_before, to_call, advice if idx == HUMAN_ID else None)
            self.update_range_from_action(idx, street, choice, to_call, amt)
            if choice == "fold":
                in_hand[idx] = False
                acted[idx] = True
            elif choice == "check":
                acted[idx] = True
            elif choice == "call":
                pay = min(self.stacks[idx], to_call)
                self.stacks[idx] -= pay
                street_contrib[idx] += pay
                acted[idx] = True
            else:
                min_inc = self.bb if current_bet == 0 else last_full_raise
                if self.stacks[idx] >= to_call + min_inc:
                    assert amt >= min_inc
                pay_target = to_call + amt
                pay = min(self.stacks[idx], pay_target)
                self.stacks[idx] -= pay
                prev_bet = current_bet
                street_contrib[idx] += pay
                acted[idx] = True
                new_bet = street_contrib[idx]
                inc = new_bet - prev_bet
                if inc > 0:
                    pre_acted = acted[:]
                    current_bet = new_bet
                    for p in range(self.n):
                        if p != idx and in_hand[p] and self.stacks[p] > 0:
                            acted[p] = False
                    full = inc >= min_inc
                    if full:
                        last_full_raise = inc
                        for p in range(self.n):
                            if in_hand[p] and self.stacks[p] > 0:
                                raise_allowed[p] = True
                    else:
                        for p in range(self.n):
                            if p != idx and in_hand[p] and self.stacks[p] > 0 and pre_acted[p]:
                                raise_allowed[p] = False
            idx = (idx + 1) % self.n

    def runout_to_river(self, deck, board):
        while len(board) < 5:
            board += deal(deck, 1)

    def award_fold_win(self, in_hand, pot):
        winner = next(i for i, a in enumerate(in_hand) if a)
        self.stacks[winner] += pot
        return winner

    def distribute_showdown(self, hands, board, total_contrib, in_hand):
        levels = sorted(set(total_contrib[i] for i, a in enumerate(in_hand) if a and total_contrib[i] > 0))
        pots = []
        last = 0
        for level in levels:
            amt = 0
            elig = []
            for i, c in enumerate(total_contrib):
                amt += max(0, min(c, level) - last)
                if in_hand[i] and c >= level:
                    elig.append(i)
            if amt:
                pots.append({"amount": amt, "eligible": elig})
            last = level
        awards = []
        for pot_obj in pots:
            amt = pot_obj["amount"]
            elig = pot_obj["eligible"]
            best = (-1, ())
            winners = []
            for pid in elig:
                v = best_hand_7(hands[pid] + board)
                if v > best:
                    best = v
                    winners = [pid]
                elif v == best:
                    winners.append(pid)
            share = amt // len(winners)
            rem = amt % len(winners)
            for w in winners:
                self.stacks[w] += share
            if rem:
                start = (self.btn + 1) % self.n
                for k in range(self.n):
                    seat = (start + k) % self.n
                    if seat in winners:
                        self.stacks[seat] += rem
                        break
            awards.append({"amount": amt, "eligible": elig, "winners": winners, "hand_class": best[0]})
        return awards

    def print_post_hand_evaluation(self):
        print("" + "=" * 78)
        print("Post-hand evaluation")
        print(f"Final board: {format_cards(self.showdown_log.get('board', []))}")
        hands = self.showdown_log.get("hands", [])
        if hands:
            for i, h in enumerate(hands):
                print(f"Seat {i} hole: {format_cards(h)}")
        print("Final stacks:", self.stacks)
        awards = self.showdown_log.get("awards", None)
        fold_winner = self.showdown_log.get("fold_winner", None)
        if fold_winner is not None:
            print(f"Hand ended by folds. Winner: Seat {fold_winner}")
        elif awards is not None:
            for j, a in enumerate(awards, 1):
                print(f"Pot {j}: {a['amount']} | Eligible {a['eligible']} | Winners {a['winners']} | Class {a['hand_class']}")
        print("Decision trace (your actions annotated when available)")
        for e in self.hand_log:
            actor = "YOU" if e["seat"] == HUMAN_ID else f"Seat {e['seat']}"
            b = format_cards(e["board"]) if e["board"] else "(preflop)"
            line = f"[{e['street']}] {actor}: {e['action']}"
            if e["action"] == "raise":
                line += f" by {e['amt']}"
            line += f" | board {b} | pot_before {e['pot_before']} | to_call {e['to_call']}"
            print(line)
            if e["seat"] == HUMAN_ID and e["advice"] is not None:
                a = e["advice"]
                rec_a, rec_amt = a["best_action"]
                rec = rec_a if rec_a != "raise" else f"raise {rec_amt}"
                print(f"  advice: {rec} | EV ≈ {a['best_ev']:.2f}")

    def next_hand(self):
        self.hand_log = []
        self.showdown_log = {}
        self.range_pct = [0.55] * self.n
        self.preflop_aggressor = None
        self.hand_no += 1
        deck = make_deck()
        self.rng.shuffle(deck)
        hands = [deal(deck, 2) for _ in range(self.n)]
        board = []
        in_hand = [True] * self.n
        street_contrib = [0] * self.n
        total_contrib = [0] * self.n
        pot = 0
        sb_pos, bb_pos = self.blind_positions()
        sb_amt = min(self.stacks[sb_pos], self.sb)
        self.stacks[sb_pos] -= sb_amt
        street_contrib[sb_pos] += sb_amt
        bb_amt = min(self.stacks[bb_pos], self.bb)
        self.stacks[bb_pos] -= bb_amt
        street_contrib[bb_pos] += bb_amt
        self.betting_round(hands, board, in_hand, street_contrib, pot, is_preflop=True)
        refund_uncalled(street_contrib, self.stacks)
        pot += sum(street_contrib)
        for i in range(self.n):
            total_contrib[i] += street_contrib[i]
        street_contrib = [0] * self.n
        if self.in_hand_count(in_hand) <= 1:
            winner = self.award_fold_win(in_hand, pot)
            self.showdown_log = {"board": board[:], "hands": hands, "fold_winner": winner}
            if self.mode in ("play", "learn"):
                self.print_post_hand_evaluation()
            self.btn = (self.btn + 1) % self.n
            return
        for to_deal in (3, 1, 1):
            board += deal(deck, to_deal)
            if self.in_hand_count(in_hand) <= 1:
                break
            if self.can_act_count(in_hand) <= 1:
                self.runout_to_river(deck, board)
                break
            self.betting_round(hands, board, in_hand, street_contrib, pot, is_preflop=False)
            refund_uncalled(street_contrib, self.stacks)
            pot += sum(street_contrib)
            for i in range(self.n):
                total_contrib[i] += street_contrib[i]
            street_contrib = [0] * self.n
            if self.in_hand_count(in_hand) <= 1:
                winner = self.award_fold_win(in_hand, pot)
                self.showdown_log = {"board": board[:], "hands": hands, "fold_winner": winner}
                if self.mode in ("play", "learn"):
                    self.print_post_hand_evaluation()
                self.btn = (self.btn + 1) % self.n
                return
        if len(board) < 5:
            self.runout_to_river(deck, board)
        awards = self.distribute_showdown(hands, board, total_contrib, in_hand)
        self.showdown_log = {"board": board[:], "hands": hands, "awards": awards}
        if self.mode in ("play", "learn"):
            self.print_post_hand_evaluation()
        self.btn = (self.btn + 1) % self.n


# helpers kept separate

def build_side_pots(total_contrib, in_hand):
    levels = sorted(set(total_contrib[i] for i, a in enumerate(in_hand) if a and total_contrib[i] > 0))
    pots = []
    last = 0
    for level in levels:
        amt = 0
        elig = []
        for i, c in enumerate(total_contrib):
            amt += max(0, min(c, level) - last)
            if in_hand[i] and c >= level:
                elig.append(i)
        if amt:
            pots.append({"amount": amt, "eligible": elig})
        last = level
    return pots


def refund_uncalled(street_contrib, stacks):
    mx = max(street_contrib)
    if mx == 0:
        return
    leaders = [i for i, c in enumerate(street_contrib) if c == mx]
    if len(leaders) != 1:
        return
    leader = leaders[0]
    second = max((c for i, c in enumerate(street_contrib) if i != leader), default=0)
    refund = mx - second
    if refund > 0:
        street_contrib[leader] -= refund
        stacks[leader] += refund


def effective_stack(self, idx, in_hand):
    opp = [self.stacks[i] for i, a in enumerate(in_hand) if a and i != idx]
    if not opp:
        return self.stacks[idx]
    return min(self.stacks[idx], max(opp))


Table.effective_stack = effective_stack


def mdf_defend_freq(self, pot_now, bet):
    if bet <= 0:
        return 1.0
    return pot_now / (pot_now + bet)


Table.mdf_defend_freq = mdf_defend_freq


if __name__ == "__main__":
    mode = MODE
    eval_mode = 'both'
    if len(sys.argv) >= 2:
        mode = sys.argv[1].strip().lower()
    if len(sys.argv) >= 3:
        eval_mode = sys.argv[2].strip().lower()
    if mode not in ("play", "learn"):
        mode = "play"
    if eval_mode not in ('human', 'mc', 'both'):
        eval_mode = 'both'
    t = Table(n=6, stack=1000, sb=5, bb=10, seed=0, mode=mode, mc_samples=MC_SAMPLES, eval_mode=eval_mode)
    print(f"Mode: {mode} | Eval mode: {eval_mode}")
    print("Controls: r=raise, c=call, f=fold, k=check")
    print("Learn mode: shows human vs MC equities. MC is parallelized.")
    while True:
        t.next_hand()
        cont = input("Next hand? (y/n): ").strip().lower()
        if cont != "y":
            break
