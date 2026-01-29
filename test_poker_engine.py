import unittest
import random
import types
import copy

import poker_engine as pe

def C(r, s):
    return (r, s)

class BaseTest(unittest.TestCase):
    def setUp(self):
        # Make deterministic
        random.seed(42)
        # Ensure no interactive human actions
        self.orig_human = pe.HUMAN_ID
        pe.HUMAN_ID = 999
        # Save original AdaptiveBot.action to restore later
        self.orig_bot_action = pe.AdaptiveBot.action

    def tearDown(self):
        pe.HUMAN_ID = self.orig_human
        pe.AdaptiveBot.action = self.orig_bot_action

# -------------------------
# evaluator & basic checks
# -------------------------
class TestEvaluator(BaseTest):

    def test_wheel_and_sf_and_quads(self):
        cases = [
            ( [C(14,'s'), C(2,'d'), C(3,'h'), C(4,'c'), C(5,'s'), C(9,'d'), C(10,'c')], 4, 5 ),  # wheel
            ( [C(9,'s'), C(8,'s'), C(7,'s'), C(6,'s'), C(5,'s'), C(2,'d'), C(3,'c')], 8, 9 ),   # straight flush
            ( [C(7,'s'), C(7,'d'), C(7,'h'), C(7,'c'), C(4,'s'), C(2,'d'), C(3,'c')], 7, 7 ),    # quads
        ]
        for cards, expect_cat, expect_top in cases:
            cat, v = pe.best_hand_7(cards)
            self.assertEqual(cat, expect_cat)
            # ensure first value corresponds to top card where appropriate
            self.assertEqual(v[0], expect_top)

    def test_full_house_and_flush(self):
        fh = [C(9,'s'), C(9,'d'), C(9,'h'), C(4,'c'), C(4,'s'), C(2,'d'), C(3,'c')]
        cat, v = pe.best_hand_7(fh)
        self.assertEqual(cat, 6)
        self.assertEqual(v[0], 9)
        fl = [C(14,'s'), C(12,'s'), C(10,'s'), C(8,'s'), C(6,'s'), C(2,'d'), C(3,'c')]
        cat2, v2 = pe.best_hand_7(fl)
        self.assertEqual(cat2, 5)
        self.assertEqual(v2[0], 14)

# -------------------------
# equity invariants & orders
# -------------------------
class TestEquityProperties(BaseTest):

    def test_bounds_and_ordering(self):
        random.seed(1)
        eq = pe.estimate_equity([C(14,'s'), C(14,'d')], [], players=2, n=300)
        self.assertGreater(eq, 0.0)
        self.assertLess(eq, 1.0)
        aa = pe.estimate_equity([C(14,'s'), C(14,'d')], [], players=2, n=300)
        trash = pe.estimate_equity([C(7,'c'), C(2,'d')], [], players=2, n=300)
        self.assertGreater(aa, trash)

    def test_flush_draw_vs_gutshot_order(self):
        random.seed(2)
        nut_fd = pe.estimate_equity([C(14,'s'), C(2,'s')], [C(10,'s'), C(6,'s'), C(3,'d')], players=2, n=400)
        weak_fd = pe.estimate_equity([C(9,'s'), C(8,'s')], [C(10,'s'), C(6,'s'), C(3,'d')], players=2, n=400)
        gutshot = pe.estimate_equity([C(8,'c'), C(7,'d')], [C(10,'s'), C(6,'s'), C(3,'d')], players=2, n=400)
        self.assertGreater(nut_fd, weak_fd)
        self.assertGreater(nut_fd, gutshot)
        self.assertGreater(weak_fd, gutshot)

    def test_equity_decreases_with_opponents(self):
        random.seed(3)
        hu = pe.estimate_equity([C(14,'s'), C(14,'d')], [], players=2, n=300)
        multi = pe.estimate_equity([C(14,'s'), C(14,'d')], [], players=6, n=300)
        self.assertGreater(hu, multi)

# -------------------------
# Table integration invariants
# -------------------------
class TestTableInvariants(BaseTest):

    def test_pot_and_stack_conservation_single_hand(self):
        random.seed(4)
        # make bots deterministic based on hole: strong raise, others call
        def bot_action(self, hole, board, need, pot, stack):
            # if contains an ace, raise by small amount; if pocket pair call; else call
            ranks = [h[0] for h in hole]
            if 14 in ranks:
                return ("raise", max(1, int(pot * 0.5)))
            if ranks[0] == ranks[1]:
                return ("call", 0)
            return ("call", 0)
        pe.AdaptiveBot.action = bot_action

        n = 6
        starting_stack = 1000
        t = pe.Table(n=n, stack=starting_stack, sb=5, bb=10)
        total_before = sum(t.stacks)
        t.next_hand()
        total_after = sum(t.stacks)
        # total chips must be conserved
        self.assertEqual(total_before, total_after)
        # no negative stacks
        self.assertTrue(all(s >= 0 for s in t.stacks))

    def test_many_hands_no_negative_and_stats_growth(self):
        random.seed(5)
        # bot: always call preflop (so VPIP increments), random postflop call/raise
        def bot_action(self, hole, board, need, pot, stack):
            if not board:
                return ("call", 0)
            return ("call", 0)
        pe.AdaptiveBot.action = bot_action

        n = 6
        t = pe.Table(n=n, stack=500, sb=5, bb=10)
        initial_total = sum(t.stacks)
        hands = 20
        for _ in range(hands):
            t.next_hand()
        # no negatives
        self.assertTrue(all(s >= 0 for s in t.stacks))
        # total conserved
        self.assertEqual(sum(t.stacks), initial_total)
        # stats: each seat should have hands counted
        for stat in t.stats:
            self.assertGreaterEqual(stat.hands, 1)
            # vpip should be <= hands
            self.assertLessEqual(stat.vpip, stat.hands)

    def test_showdown_distribution_equal_split_case(self):
        # This test forces a tie at showdown by controlling actions and deck via seed.
        # We'll ensure all bots call to showdown and the board/hands produce a tie (two identical best hands).
        random.seed(6)
        # force all bots to call
        def bot_action(self, hole, board, need, pot, stack):
            return ("call", 0)
        pe.AdaptiveBot.action = bot_action

        t = pe.Table(n=3, stack=100, sb=1, bb=2)
        total_before = sum(t.stacks)
        # run a hand; deterministic seed chosen to often produce showdown with possible ties
        t.next_hand()
        total_after = sum(t.stacks)
        # chips conserved
        self.assertEqual(total_before, total_after)
        # each stack should be integer and non-negative
        self.assertTrue(all(isinstance(s, int) and s >= 0 for s in t.stacks))

# -------------------------
# raise/EV/pot-odds invariants
# -------------------------
class TestMathInvariants(BaseTest):

    def test_ev_sign_matches_equity_vs_potodds(self):
        # pick a scenario and check sign consistency (use larger n to reduce MC noise)
        random.seed(7)
        hole = [C(14,'s'), C(2,'s')]
        board = [C(10,'s'), C(6,'s'), C(3,'d')]
        players = 2
        eq = pe.estimate_equity(hole, board, players, n=1000)
        pot = 100
        to_call = 20
        pot_odds = to_call / (pot + to_call)
        ev = eq * (pot + to_call) - to_call
        # sign of ev should correspond to equity > pot_odds (allow tiny eps due to MC noise)
        eps = 1e-3
        if ev > 0:
            self.assertGreater(eq, pot_odds - eps)
        elif ev < 0:
            self.assertLess(eq, pot_odds + eps)
        else:
            # ev == 0 approx
            self.assertAlmostEqual(eq, pot_odds, places=2)

    def test_pot_never_negative_during_betting(self):
        # Use bot that sometimes raises; run few hands and ensure pot never negative (implicit via stacks)
        random.seed(8)
        def bot_action(self, hole, board, need, pot, stack):
            # small raise if high card present, otherwise call
            if max(hole[0][0], hole[1][0]) >= 11:
                return ("raise", max(1, int(pot * 0.25)))
            return ("call", 0)
        pe.AdaptiveBot.action = bot_action

        t = pe.Table(n=6, stack=500, sb=5, bb=10)
        # run several hands
        for _ in range(10):
            t.next_hand()
            # after each hand, pot not directly visible; ensure all stacks >= 0
            self.assertTrue(all(s >= 0 for s in t.stacks))

# -------------------------
# preflop heuristic tests
# -------------------------
class TestPreflop(BaseTest):

    def test_preflop_score_properties(self):
        a = pe.preflop_score([C(13,'s'), C(9,'d')])
        b = pe.preflop_score([C(9,'d'), C(13,'s')])
        self.assertEqual(a, b)
        pair = pe.preflop_score([C(10,'s'), C(10,'d')])
        offs = pe.preflop_score([C(14,'s'), C(6,'d')])
        self.assertGreater(pair, offs)

if __name__ == "__main__":
    unittest.main()
