# poker_engine.py
import itertools
import random
from collections import Counter

RANKS = list(range(2, 15))
SUITS = "cdhs"
HUMAN_ID = 0


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
            r = desc[:]
            if 14 in r:
                r.append(1)
            for i in range(len(r) - 4):
                if r[i] - r[i + 4] == 4:
                    return r[i]
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
    top = [i for i, c in enumerate(street_contrib) if c == mx]
    if len(top) != 1:
        return
    top_i = top[0]
    second = max((c for i, c in enumerate(street_contrib) if i != top_i), default=0)
    refund = mx - second
    if refund > 0:
        street_contrib[top_i] -= refund
        stacks[top_i] += refund


class AdaptiveBot:
    def action(self, hole, board, to_call, pot, stack, valid_actions):
        if to_call == 0 and "check" in valid_actions:
            return "check", 0
        if to_call > 0 and "call" in valid_actions:
            return "call", 0
        return "fold", 0


class Table:
    def __init__(self, n=6, stack=1000, sb=5, bb=10, seed=0, bot=None):
        self.n = n
        self.stacks = [stack] * n
        self.sb = sb
        self.bb = bb
        self.btn = 0
        self.rng = random.Random(seed)
        self.bot = bot or AdaptiveBot()
        self.total_chips = sum(self.stacks)

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

    def betting_round(self, hands, board, in_hand, street_contrib, pot, is_preflop):
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

            if idx == HUMAN_ID:
                choice = input(f"Action {valid}: ").strip().lower()
                if choice == "r":
                    choice = "raise"
                if choice == "c":
                    choice = "call"
                if choice == "f":
                    choice = "fold"
                if choice == "k":
                    choice = "check"
                amt = 0
                if choice == "raise":
                    amt = int(input("Raise by: "))
            else:
                choice, amt = self.bot.action(
                    hands[idx],
                    board,
                    to_call,
                    pot + sum(street_contrib),
                    self.stacks[idx],
                    valid,
                )

            valid.index(choice)

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

    def distribute_showdown(self, hands, board, total_contrib, in_hand):
        pots = build_side_pots(total_contrib, in_hand)
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

    def next_hand(self):
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
            self.award_fold_win(in_hand, pot)
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
                self.award_fold_win(in_hand, pot)
                self.btn = (self.btn + 1) % self.n
                return

        if len(board) < 5:
            self.runout_to_river(deck, board)

        self.distribute_showdown(hands, board, total_contrib, in_hand)
        self.btn = (self.btn + 1) % self.n

    def simulate(self, hands=1000):
        for _ in range(hands):
            self.next_hand()
            self.check_invariants()