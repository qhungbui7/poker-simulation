# poker_engine_fixed.py
import random
import itertools
from collections import Counter

RANKS = list(range(2, 15))
SUITS = "cdhs"

HUMAN_ID = 0
MODE = "learn"
BET_SIZES = [0.33, 0.66, 1.0]
N_MC = 500

def format_card(card):
    r, s = card
    m = {11: "J", 12: "Q", 13: "K", 14: "A"}
    sm = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
    return f"{m.get(r, r)}{sm.get(s, s)}"

def make_deck():
    return [(r, s) for r in RANKS for s in SUITS]

def deal(deck, n):
    return [deck.pop() for _ in range(n)]

def ranks(cards):
    return [r for r, _ in cards]

def best_hand_7(cards):
    def eval5(c):
        rc = Counter(ranks(c))
        sc = Counter(s for _, s in c)
        uniq_desc = sorted(set(ranks(c)), reverse=True)
        def find_straight_high(sorted_desc):
            r = sorted_desc[:]
            if 14 in r:
                r.append(1)
            for i in range(len(r) - 4):
                if r[i] - r[i + 4] == 4:
                    return r[i]
            return None
        flush_suit = next((s for s, cnt in sc.items() if cnt >= 5), None)
        if flush_suit:
            flush_ranks = sorted(set(r for r, s in c if s == flush_suit), reverse=True)
            sf_high = find_straight_high(flush_ranks)
            if sf_high:
                return (8, (sf_high,))
        straight_high = find_straight_high(uniq_desc)
        counts = sorted(rc.items(), key=lambda x: (x[1], x[0]), reverse=True)
        if counts[0][1] == 4:
            four = counts[0][0]
            kicker = max(r for r in uniq_desc if r != four)
            return (7, (four, kicker))
        if counts[0][1] == 3 and any(cnt >= 2 for _, cnt in counts[1:]):
            three = counts[0][0]
            pair = next(r for r, cnt in counts[1:] if cnt >= 2)
            return (6, (three, pair))
        if flush_suit:
            top5 = tuple(sorted([r for r, s in c if s == flush_suit], reverse=True)[:5])
            return (5, top5)
        if straight_high:
            return (4, (straight_high,))
        if counts[0][1] == 3:
            trips = counts[0][0]
            kickers = tuple([r for r in uniq_desc if r != trips][:2])
            return (3, (trips,) + kickers)
        if len(counts) >= 2 and counts[0][1] == 2 and counts[1][1] == 2:
            p1, p2 = counts[0][0], counts[1][0]
            kicker = next(r for r in uniq_desc if r not in (p1, p2))
            return (2, (p1, p2, kicker))
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

def estimate_equity(hole, board, players, n=N_MC):
    wins = 0.0
    known = set(hole + board)
    remaining = [(r, s) for r in RANKS for s in SUITS if (r, s) not in known]
    need_cards = 2 * (players - 1) + (5 - len(board))
    for _ in range(n):
        draw = random.sample(remaining, need_cards)
        opps = [draw[i*2:(i+1)*2] for i in range(players - 1)]
        full_board = board + draw[2*(players-1):]
        hero = best_hand_7(hole + full_board)
        best = hero
        tie = 1
        for o in opps:
            v = best_hand_7(o + full_board)
            if v > best:
                best = v
                tie = 0
            elif v == best:
                tie += 1
        if best == hero:
            wins += 1.0 / tie
    return wins / n

def get_hand_name(v):
    d = {8: "Straight Flush",7: "Four of a Kind",6: "Full House",5: "Flush",4: "Straight",3: "Three of a Kind",2: "Two Pair",1: "One Pair",0: "High Card"}
    return d.get(v[0], "Unknown")

def pos_label(btn, seat, n):
    rel = (seat - btn) % n
    if rel == 0: return "BTN"
    if rel == 1: return "SB"
    if rel == 2: return "BB"
    order = ["UTG", "UTG+1", "MP", "CO", "CO+1", "HJ"]
    return order[(rel-3) % len(order)]

def preflop_score(hole):
    s = hole[0][0] + hole[1][0]
    if hole[0][0] == hole[1][0]:
        s += 15
    return s

def rule_action(hole, board, to_call, pot, stack):
    if not board:
        s = preflop_score(hole)
        if s > 30: return "raise", int(pot * 0.66)
        if s > 20: return "call", 0
        return "fold", 0
    cat = best_hand_7(hole + board)[0]
    if cat >= 4: return "raise", int(pot * 0.66)
    if cat >= 2: return "call", 0
    return "fold", 0

def explain_decision(hole, board, to_call, pot, stack, players):
    print("\n--- Decision helper ---")
    if not board:
        s = preflop_score(hole)
        print("Preflop", f"Hand: {format_card(hole[0])} {format_card(hole[1])}", f"Score:{s}")
    else:
        hv = best_hand_7(hole+board)
        print("Postflop", f"Hand: {format_card(hole[0])} {format_card(hole[1])}")
        print("Board:", " ".join(format_card(c) for c in board))
        print("Best:", get_hand_name(hv), "Category", hv[0])
        eq = estimate_equity(hole, board, players)
        print(f"Equity ≈ {eq:.2%}")
    print("---")

class Stats:
    def __init__(self):
        self.hands = 0
        self.vpip = 0
        self.pfr = 0
        self.aggr = 0
        self.calls = 0
    def vpip_rate(self): return self.vpip / self.hands if self.hands else 0
    def pfr_rate(self): return self.pfr / self.hands if self.hands else 0
    def af(self): return self.aggr / self.calls if self.calls else 0.0

class AdaptiveBot:
    def __init__(self, stats): self.stats = stats
    def action(self, hole, board, need, pot, stack):
        vpip = self.stats[HUMAN_ID].vpip_rate()
        pfr = self.stats[HUMAN_ID].pfr_rate()
        loose = vpip > 0.4
        passive = pfr < 0.15
        if not board:
            s = preflop_score(hole)
            if loose: s += 5
            if s > 28: return "raise", int(pot * 0.7)
            if s > 20: return "call", 0
            return "fold", 0
        cat = best_hand_7(hole + board)[0]
        if cat >= 4: return "raise", int(pot * (1.0 if loose else 0.66))
        if cat >= 2: return "call", 0
        if passive and need == 0: return "raise", int(pot * 0.33)
        return "fold", 0

def human_action(hole, board, to_call, pot, stack, players, btn, seat, n):
    pl = pos_label(btn, seat, n)
    print(f"\nYou are {pl} Seat {seat} Stack={stack}")
    print("Your:", format_card(hole[0]), format_card(hole[1]))
    if MODE == "learn": explain_decision(hole, board, to_call, pot, stack, players)
    elif MODE in ("hint","eval"): print("Hint:", rule_action(hole, board, to_call, pot, stack)[0])
    if MODE == "eval":
        eq = estimate_equity(hole, board, players)
        odds = to_call / (pot + to_call) if to_call else 0
        print(f"Equity≈{eq:.2f} PotOdds≈{odds:.2f}")
    while True:
        a = input("Action [f/c/r]: ").strip().lower()
        if a in ("f","c","r"): break
        print("Invalid")
    if a == "r":
        print("Sizes:", BET_SIZES)
        s = float(input("Size: ").strip())
        if s not in BET_SIZES: raise ValueError(f"Size must be one of {BET_SIZES}")
        amt = int(pot * s)
        return "raise", max(1, amt)
    if a == "c": return "call", 0
    return "fold", 0

class Table:
    def __init__(self, n=6, stack=1000, sb=5, bb=10):
        self.n = n
        self.stacks = [stack] * n
        self.btn = 0
        self.sb = sb
        self.bb = bb
        self.stats = [Stats() for _ in range(n)]
        self.bot = AdaptiveBot(self.stats)
    def show_table_state(self, hands, contrib, active, pot):
        print("\n-- Table snapshot --")
        for i in range(self.n):
            p = pos_label(self.btn, i, self.n)
            hand_str = "??" if active[i] and i != HUMAN_ID else " ".join(format_card(c) for c in hands[i])
            print(f"Seat {i:>2} {p:>6} | Stack {self.stacks[i]:>5} | Contrib {contrib[i]:>4} | Active {active[i]} | Cards: {hand_str}")
        print(f"Pot: {pot}")
    def next_hand(self):
        deck = make_deck()
        random.shuffle(deck)
        hands = [deal(deck,2) for _ in range(self.n)]
        board = []
        active = [True]*self.n
        contrib = [0]*self.n
        pot = 0
        sb_i = (self.btn + 1) % self.n
        bb_i = (self.btn + 2) % self.n
        self.stacks[sb_i] -= self.sb
        self.stacks[bb_i] -= self.bb
        contrib[sb_i] = self.sb
        contrib[bb_i] = self.bb
        pot = self.sb + self.bb
        to_call = self.bb
        last_raise_by = self.bb
        last_raiser = None
        action_history = []
        def betting(start, is_preflop=False):
            nonlocal pot, to_call, last_raise_by, last_raiser
            if sum(active) <= 1: return
            max_contrib = max(contrib)
            acted = [False] * self.n
            i = start
            while True:
                if not active[i] or self.stacks[i] <= 0:
                    acted[i] = True
                else:
                    need = max_contrib - contrib[i]
                    if i == HUMAN_ID:
                        act, amt = human_action(hands[i], board, need, pot, self.stacks[i], sum(active), self.btn, i, self.n)
                    else:
                        act, amt = self.bot.action(hands[i], board, need, pot, self.stacks[i])
                    paid = 0
                    if act == "fold":
                        active[i] = False
                        acted[i] = True
                        action_history.append((i, pos_label(self.btn, i, self.n), "fold", 0, contrib[i]))
                    elif act == "call":
                        pay = min(need, self.stacks[i])
                        if need > 0:
                            pay = min(need, self.stacks[i])
                            self.stacks[i] -= pay
                            contrib[i] += pay
                            pot += pay
                        else:
                            pay = 0

                        self.stats[i].calls += 1
                        if is_preflop: self.stats[i].vpip += 1
                        acted[i] = True
                        action_history.append((i, pos_label(self.btn, i, self.n), "call", pay, contrib[i]))
                    elif act == "raise":
                        raise_by = amt
                        min_raise_by = max(last_raise_by, self.bb) if is_preflop else max(last_raise_by, 1)
                        if raise_by < min_raise_by: raise_by = min_raise_by
                        need = max_contrib - contrib[i]
                        target_pay = need + raise_by
                        pay = min(target_pay, self.stacks[i])
                        actual_raise_by = pay - need
                        if pay > 0:
                            self.stacks[i] -= pay
                            contrib[i] += pay
                            pot += pay
                        max_contrib = contrib[i]
                        to_call = max_contrib
                        if actual_raise_by > 0: last_raise_by = actual_raise_by
                        last_raiser = i
                        acted = [False] * self.n
                        acted[i] = True
                        self.stats[i].aggr += 1
                        if is_preflop:
                            self.stats[i].vpip += 1
                            self.stats[i].pfr += 1
                        action_history.append((i, pos_label(self.btn, i, self.n), f"raise_by({actual_raise_by})", pay, contrib[i]))
                i = (i + 1) % self.n
                if sum(active) <= 1: break
                if all((not active[j]) or acted[j] for j in range(self.n)) and all((not active[j]) or contrib[j] == max_contrib for j in range(self.n)):
                    break

        self.show_table_state(hands, contrib, active, pot)
        betting((bb_i + 1) % self.n, is_preflop=True)
        if sum(active) > 1:
            board += deal(deck, 3)
            betting((self.btn + 1) % self.n)
        if sum(active) > 1:
            board += deal(deck, 1)
            betting((self.btn + 1) % self.n)
        if sum(active) > 1:
            board += deal(deck, 1)
            betting((self.btn + 1) % self.n)
        best = None
        winners = []
        showdown_info = []
        for i in range(self.n):
            if active[i]:
                v = best_hand_7(hands[i] + board)
                showdown_info.append((i, get_hand_name(v), v, hands[i]))
                if best is None or v > best:
                    best = v
                    winners = [i]
                elif v == best:
                    winners.append(i)
        pot_per = pot // len(winners)
        rem = pot % len(winners)
        for idx, w in enumerate(winners):
            self.stacks[w] += pot_per + (1 if idx < rem else 0)
        for i in range(self.n):
            self.stats[i].hands += 1
        self.btn = (self.btn + 1) % self.n
        print("\n=== HAND SUMMARY ===")
        print("Button:", (self.btn-1) % self.n, "| SB:", sb_i, "BB:", bb_i)
        print("Board:", " ".join(format_card(c) for c in board) if board else "[]")
        print("\nAction history:")
        for act in action_history:
            seat, pos, a, paid, contrib_after = act
            print(f" Seat {seat:>2} {pos:>6} | {a:>20} | paid {paid:>4} | contrib {contrib_after}")
        print("\nShowdown (active players):")
        for i, name, val, hs in showdown_info:
            print(f" Seat {i:>2} {pos_label(self.btn-1, i, self.n):>6} | {format_card(hs[0])} {format_card(hs[1])} | {name} | val {val}")
        print("\nWinners:", winners)
        print("Stacks:", self.stacks)
        print("\nStats (VPIP/PFR):")
        for i in range(self.n):
            print(f" Seat {i:>2} | VPIP {self.stats[i].vpip_rate():.2%} | PFR {self.stats[i].pfr_rate():.2%} | Hands {self.stats[i].hands}")
        print("====================\n")

if __name__ == "__main__":
    t = Table()
    while True:
        t.next_hand()
        if input("Next hand? [enter/q]: ").strip().lower() == "q": break
