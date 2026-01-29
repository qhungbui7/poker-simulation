import random
import itertools
from collections import Counter

RANKS = list(range(2, 15))
SUITS = "cdhs"

HUMAN_ID = 0
MODE = "eval"
BET_SIZES = [0.33, 0.66, 1.0]
N_MC = 500

def format_card(card):
    """Convert card tuple to human-readable string."""
    rank, suit = card
    rank_map = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    rank_str = rank_map.get(rank, str(rank))
    suit_map = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}
    return f"{rank_str}{suit_map.get(suit, suit)}"

def preflop_strength(hole):
    """Calculate preflop hand strength score."""
    s = hole[0][0] + hole[1][0]
    if hole[0][0] == hole[1][0]:
        s += 15
    return s

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
        uniq = sorted(set(ranks(c)), reverse=True)
        # Add Ace low for wheel detection
        uniq_with_ace_low = uniq + ([1] if 14 in uniq else [])
        
        # Detect straight (including wheel A-2-3-4-5)
        straight = None
        for i in range(len(uniq_with_ace_low) - 4):
            if uniq_with_ace_low[i] - uniq_with_ace_low[i + 4] == 4:
                straight = uniq_with_ace_low[i]
                break
        
        # Detect flush and get flush suit
        flush_suit = None
        for suit, count in sc.items():
            if count >= 5:
                flush_suit = suit
                break
        
        flush = flush_suit is not None
        
        # Get flush cards if flush exists
        flush_cards = sorted([r for r, s in c if s == flush_suit], reverse=True)[:5] if flush else None
        
        counts = sorted(rc.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        if flush and straight:
            return (8, straight)
        if counts[0][1] == 4:
            return (7, counts[0][0])
        if counts[0][1] == 3 and counts[1][1] >= 2:
            return (6, counts[0][0])
        if flush:
            return (5, flush_cards)
        if straight:
            return (4, straight)
        if counts[0][1] == 3:
            return (3, counts[0][0])
        if counts[0][1] == 2 and counts[1][1] == 2:
            return (2, (counts[0][0], counts[1][0]))
        if counts[0][1] == 2:
            return (1, counts[0][0])
        return (0, uniq[:5])

    best = (-1, None)
    for c in itertools.combinations(cards, 5):
        v = eval5(c)
        if v > best:
            best = v
    return best

def estimate_equity(hole, board, players, n=N_MC):
    wins = 0
    known = hole + board
    for _ in range(n):
        deck = make_deck()
        for c in known:
            deck.remove(c)
        random.shuffle(deck)
        opps = [deal(deck, 2) for _ in range(players - 1)]
        full = board + deal(deck, 5 - len(board))
        hero = best_hand_7(hole + full)
        best = hero
        tie = 1
        for o in opps:
            v = best_hand_7(o + full)
            if v > best:
                best = v
                tie = 0
            elif v == best:
                tie += 1
        if best == hero:
            wins += 1 / tie
    return wins / n

class Stats:
    def __init__(self):
        self.hands = 0
        self.vpip = 0
        self.pfr = 0
        self.aggr = 0
        self.calls = 0

    def vpip_rate(self):
        return self.vpip / self.hands if self.hands else 0

    def pfr_rate(self):
        return self.pfr / self.hands if self.hands else 0

    def af(self):
        return self.aggr / self.calls if self.calls else 0.0

class AdaptiveBot:
    def __init__(self, stats):
        self.stats = stats

    def action(self, hole, board, to_call, pot, stack):
        vpip = self.stats[HUMAN_ID].vpip_rate()
        pfr = self.stats[HUMAN_ID].pfr_rate()
        loose = vpip > 0.4
        passive = pfr < 0.15

        cat = best_hand_7(hole + board)[0] if board else None

        if not board:
            s = preflop_strength(hole)
            if loose:
                s += 5
            if s > 28:
                return "raise", int(pot * 0.7)
            if s > 20:
                return "call", 0
            return "fold", 0

        if cat >= 4:
            return "raise", int(pot * (1.0 if loose else 0.66))
        if cat >= 2:
            return "call", 0
        if passive and to_call == 0:
            return "raise", int(pot * 0.33)
        return "fold", 0

def human_action(hole, board, to_call, pot, stack, players):
    print("\nYour hand:", [format_card(c) for c in hole])
    print("Board:", [format_card(c) for c in board] if board else "[]")
    print(f"Pot={pot} ToCall={to_call} Stack={stack}")
    if MODE in ("hint", "eval"):
        print("Suggested:", rule_action(hole, board, to_call, pot, stack)[0])
    if MODE == "eval":
        eq = estimate_equity(hole, board, players)
        odds = to_call / (pot + to_call) if to_call else 0
        print(f"Equity≈{eq:.2f} PotOdds≈{odds:.2f}")
    
    while True:
        a = input("Action [f/c/r]: ").strip().lower()
        if a in ("f", "c", "r"):
            break
        print("Invalid action. Please enter f (fold), c (call), or r (raise).")
    
    if a == "r":
        print("Sizes:", BET_SIZES)
        while True:
            try:
                s = float(input("Size: "))
                if s < 0:
                    print("Size must be non-negative.")
                    continue
                if s not in BET_SIZES:
                    print(f"Size should be one of {BET_SIZES}")
                return "raise", int(pot * s)
            except ValueError:
                print("Invalid input. Please enter a number.")
    if a == "c":
        return "call", 0
    return "fold", 0

def rule_action(hole, board, to_call, pot, stack):
    if not board:
        s = preflop_strength(hole)
        if s > 30:
            return "raise", int(pot * 0.66)
        if s > 20:
            return "call", 0
        return "fold", 0
    cat = best_hand_7(hole + board)[0]
    if cat >= 4:
        return "raise", int(pot * 0.66)
    if cat >= 2:
        return "call", 0
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

    def next_hand(self):
        deck = make_deck()
        random.shuffle(deck)
        hands = [deal(deck, 2) for _ in range(self.n)]
        board = []
        active = [True] * self.n
        contrib = [0] * self.n
        pot = 0

        sb_i = (self.btn + 1) % self.n
        bb_i = (self.btn + 2) % self.n

        self.stacks[sb_i] -= self.sb
        self.stacks[bb_i] -= self.bb
        contrib[sb_i] = self.sb
        contrib[bb_i] = self.bb
        pot = self.sb + self.bb
        to_call = self.bb

        def betting(start, is_preflop=False):
            nonlocal pot, to_call
            if sum(active) <= 1:
                return
            i = start
            last = None
            while True:
                if active[i] and self.stacks[i] > 0:
                    need = to_call - contrib[i]
                    if i == HUMAN_ID:
                        act, amt = human_action(hands[i], board, need, pot, self.stacks[i], sum(active))
                    else:
                        act, amt = self.bot.action(hands[i], board, need, pot, self.stacks[i])
                    if act == "fold":
                        active[i] = False
                    elif act == "call":
                        pay = min(need, self.stacks[i])
                        self.stacks[i] -= pay
                        contrib[i] += pay
                        pot += pay
                        self.stats[i].calls += 1
                        if is_preflop:
                            self.stats[i].vpip += 1
                    elif act == "raise":
                        # Validate minimum raise size (must be at least the current to_call amount)
                        min_raise = max(to_call, self.bb) if is_preflop else to_call
                        if amt < min_raise:
                            amt = min_raise
                        pay = min(need + amt, self.stacks[i])
                        self.stacks[i] -= pay
                        contrib[i] += pay
                        pot += pay
                        # Update to_call to the new amount others must call
                        to_call = contrib[i]
                        # If this is a full raise (not all-in), reopen action
                        if pay == need + amt:
                            last = i
                        self.stats[i].aggr += 1
                        if is_preflop:
                            self.stats[i].vpip += 1
                            self.stats[i].pfr += 1
                i = (i + 1) % self.n
                if i == start and last is None:
                    break
                if i == last:
                    last = None
                if sum(active) <= 1:
                    break

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
        for i in range(self.n):
            if active[i]:
                v = best_hand_7(hands[i] + board)
                if best is None or v > best:
                    best = v
                    winners = [i]
                elif v == best:
                    winners.append(i)

        pot_per_winner = pot // len(winners)
        remainder = pot % len(winners)
        for idx, w in enumerate(winners):
            self.stacks[w] += pot_per_winner + (1 if idx < remainder else 0)

        for i in range(self.n):
            self.stats[i].hands += 1

        self.btn = (self.btn + 1) % self.n
        print("\nBoard:", [format_card(c) for c in board] if board else "[]")
        print("Winners:", winners)
        print("Stacks:", self.stacks)

if __name__ == "__main__":
    t = Table()
    while True:
        t.next_hand()
        if input("\nNext hand? [enter/q]: ") == "q":
            break
