# test_poker_engine.py
import unittest
import poker_engine as pe


class TestPokerEngine(unittest.TestCase):
    def setUp(self):
        self.original_human_id = pe.HUMAN_ID
        pe.HUMAN_ID = -1

    def tearDown(self):
        pe.HUMAN_ID = self.original_human_id

    # --- Side pots ---
    def test_side_pot_logic_basic(self):
        total = [100, 200, 200, 10]
        in_hand = [True, True, True, False]
        pots = pe.build_side_pots(total, in_hand)

        self.assertEqual(len(pots), 2)
        self.assertEqual(pots[0]["amount"], 310)
        self.assertEqual(sorted(pots[0]["eligible"]), [0, 1, 2])
        self.assertEqual(pots[1]["amount"], 200)
        self.assertEqual(sorted(pots[1]["eligible"]), [1, 2])

    # --- Uncalled refund (ENGINE SIGNATURE: 2 args) ---
    def test_refund_uncalled(self):
        street = [50, 200]
        stacks = [1000, 1000]
        pe.refund_uncalled(street, stacks)
        self.assertEqual(street, [50, 50])
        self.assertEqual(stacks, [1000, 1150])

    def test_refund_uncalled_on_fold_spot(self):
        street = [150, 50, 50]
        stacks = [0, 0, 0]
        pe.refund_uncalled(street, stacks)
        self.assertEqual(street, [50, 50, 50])
        self.assertEqual(stacks, [100, 0, 0])

    def test_refund_uncalled_no_refund_when_tied_max(self):
        street = [100, 100, 50]
        stacks = [0, 0, 0]
        pe.refund_uncalled(street, stacks)
        self.assertEqual(street, [100, 100, 50])
        self.assertEqual(stacks, [0, 0, 0])

    # --- Evaluator ---
    def test_evaluator_wheel_straight(self):
        # A-2-3-4-5 straight should be 5-high
        cards = [(14, "s"), (2, "d"), (3, "c"), (4, "h"), (5, "s"), (13, "d"), (12, "c")]
        rank, kickers = pe.best_hand_7(cards)
        self.assertEqual(rank, 4)
        self.assertEqual(kickers, (5,))

    def test_evaluator_wheel_straight_flush(self):
        # A-2-3-4-5 all same suit => straight flush, 5-high
        cards = [(14, "s"), (2, "s"), (3, "s"), (4, "s"), (5, "s"), (13, "d"), (12, "c")]
        rank, kickers = pe.best_hand_7(cards)
        self.assertEqual(rank, 8)
        self.assertEqual(kickers, (5,))

    # --- Betting round ---
    def test_betting_round_standard(self):
        t = pe.Table(n=3, stack=1000, seed=0)
        hands = [[(2, "s"), (3, "s")]] * 3
        in_hand = [True, True, True]
        contrib = [0, 0, 0]

        moves = iter(
            [
                ("check", 0),
                ("check", 0),
                ("raise", 100),
                ("call", 0),
                ("call", 0),
            ]
        )
        t.bot.action = lambda *args: next(moves)

        t.betting_round(hands, [], in_hand, contrib, pot=0, is_preflop=False)
        self.assertEqual(contrib, [100, 100, 100])
        self.assertRaises(StopIteration, next, moves)

    def test_betting_round_all_in_call_terminates(self):
        t = pe.Table(n=2, stack=1000, seed=0)
        t.stacks[1] = 50

        hands = [[(14, "s"), (14, "h")]] * 2
        in_hand = [True, True]
        contrib = [0, 0]

        moves = iter([("check", 0), ("raise", 100), ("call", 0)])
        t.bot.action = lambda *args: next(moves)

        t.betting_round(hands, [], in_hand, contrib, pot=0, is_preflop=False)
        self.assertEqual(t.stacks[1], 0)
        self.assertEqual(contrib[1], 50)
        self.assertEqual(contrib[0], 100)
        self.assertRaises(StopIteration, next, moves)

    def test_fold_sequence_leaves_one_in_hand(self):
        t = pe.Table(n=3, stack=1000, seed=0)
        hands = [[(2, "s"), (3, "s")]] * 3
        in_hand = [True, True, True]
        contrib = [0, 0, 0]

        moves = iter([("raise", 50), ("fold", 0), ("fold", 0)])
        t.bot.action = lambda *args: next(moves)

        t.betting_round(hands, [], in_hand, contrib, pot=0, is_preflop=False)
        self.assertEqual(in_hand, [False, True, False])
        self.assertEqual(contrib, [0, 50, 0])
        self.assertRaises(StopIteration, next, moves)

    def test_heads_up_positions(self):
        t = pe.Table(n=2, stack=100, sb=5, bb=10, seed=0)

        t.btn = 0
        self.assertEqual(t.blind_positions(), (0, 1))
        self.assertEqual(t.first_to_act(True), 0)
        self.assertEqual(t.first_to_act(False), 1)

        t.btn = 1
        self.assertEqual(t.blind_positions(), (1, 0))
        self.assertEqual(t.first_to_act(True), 1)
        self.assertEqual(t.first_to_act(False), 0)

    def test_incomplete_raise_does_not_reopen_for_pre_actors(self):
        t = pe.Table(n=3, stack=1000, bb=10, seed=0)
        t.stacks[2] = 130

        hands = [[(2, "s"), (3, "s")]] * 3
        in_hand = [True, True, True]
        contrib = [0, 0, 0]

        step = {"i": 0}

        def spy_action(hole, board, to_call, pot, stack, valid):
            step["i"] += 1
            if step["i"] == 1:
                return "raise", 100
            if step["i"] == 2:
                return "raise", 30
            if step["i"] == 3:
                return "call", 0
            if step["i"] == 4:
                self.assertNotIn("raise", valid)
                return "call", 0
            raise StopIteration

        t.bot.action = spy_action
        t.betting_round(hands, [], in_hand, contrib, pot=0, is_preflop=False)
        self.assertEqual(contrib, [130, 130, 130])

    def test_no_betting_when_only_one_can_act(self):
        t = pe.Table(n=3, stack=1000, seed=0)
        t.stacks[1] = 0
        t.stacks[2] = 0

        hands = [[(2, "s"), (3, "s")]] * 3
        in_hand = [True, True, True]
        contrib = [0, 0, 0]

        called = {"v": False}

        def should_not_be_called(*args):
            called["v"] = True
            return "check", 0

        t.bot.action = should_not_be_called
        t.betting_round(hands, [], in_hand, contrib, pot=0, is_preflop=False)
        self.assertFalse(called["v"])

    # --- Coach helpers ---
    def test_board_texture_connected(self):
        board = [(6, "c"), (7, "d"), (8, "h")]
        tex = pe.board_texture(board)
        self.assertEqual(tex["summary"], "connected")
        self.assertEqual(tex["favours"], "caller / wide ranges")

    def test_board_texture_high_card(self):
        board = [(14, "c"), (13, "d"), (7, "h")]
        tex = pe.board_texture(board)
        self.assertEqual(tex["summary"], "high-card")
        self.assertEqual(tex["favours"], "preflop aggressor")

    def test_mdf_formula(self):
        t = pe.Table(n=2, stack=1000, seed=0)
        self.assertAlmostEqual(t.mdf_defend_freq(100, 50), 100 / 150)

    def test_blocker_hints_flush_ace(self):
        board = [(2, "s"), (9, "s"), (13, "s")]
        hole = [(14, "s"), (4, "d")]
        b = pe.blocker_hints(hole, board)
        self.assertIsNotNone(b["flush_blocker"])
        self.assertIn("Blocks nut flush", b["note"])


if __name__ == "__main__":
    unittest.main()