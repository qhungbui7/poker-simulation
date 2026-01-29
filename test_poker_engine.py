import unittest
import poker_engine as pe

class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        # FIX: Disable human interaction for ALL tests.
        # Setting this to -1 ensures the engine always uses 'self.bot.action'
        # instead of asking for input(), allowing us to mock every player.
        self.original_human_id = pe.HUMAN_ID
        pe.HUMAN_ID = -1

    def tearDown(self):
        # Restore original state
        pe.HUMAN_ID = self.original_human_id

    # --- A. Test Side Pot Construction ---
    def test_side_pot_logic_basic(self):
        """
        Verifies that side pots are correctly created when players have unequal stacks.
        Scenario:
        - P0: 100 (All-in)
        - P1: 200 (All-in)
        - P2: 500 (Covering, puts in 200)
        - P3: Folded (put in 10)
        Result should be two pots: Main (310) and Side (200).
        """
        contrib = [100, 200, 200, 10]
        active = [True, True, True, False]
        
        pots = pe.build_side_pots(contrib, active)
        
        self.assertEqual(len(pots), 2)
        
        # Pot 1 (Main): Everyone contributed at least 100 (or folded 10)
        # Calculation: P0(100)+P1(100)+P2(100)+P3(10) = 310
        self.assertEqual(pots[0]['amount'], 310)
        self.assertEqual(sorted(pots[0]['eligible']), [0, 1, 2])
        
        # Pot 2 (Side): Chips above 100
        # Calculation: P1(100)+P2(100) = 200
        self.assertEqual(pots[1]['amount'], 200)
        self.assertEqual(sorted(pots[1]['eligible']), [1, 2])

    def test_side_pot_uncalled_return(self):
        """
        Verifies behavior when a raise is uncalled (effectively a single-player side pot).
        """
        contrib = [50, 200]
        active = [True, True]
        
        pots = pe.build_side_pots(contrib, active)
        
        # Main Pot: 50 from each = 100 total
        self.assertEqual(pots[0]['amount'], 100)
        
        # Side Pot: 150 from P1, only P1 eligible
        self.assertEqual(pots[1]['amount'], 150)
        self.assertEqual(pots[1]['eligible'], [1]) 

    # --- B. Test Betting State Machine ---
    def test_betting_termination_standard(self):
        """
        Tests a standard betting round with Check/Raise/Call sequence.
        Since HUMAN_ID is -1, we can script Seat 0's actions via the mock.
        """
        t = pe.Table(n=3, stack=1000)
        hands = [[(2, 's'), (3, 's')]] * 3
        board = []
        active = [True, True, True]
        contrib = [0, 0, 0]
        
        # Scripted Sequence:
        # P1: Check
        # P2: Check
        # P0: Raise 100
        # P1: Call
        # P2: Call
        # End of round (everyone matched 100)
        
        moves = iter([
            ("check", 0),  # P1 action
            ("check", 0),  # P2 action
            ("raise", 100),# P0 action (Now handled by mock!)
            ("call", 0),   # P1 calls raise
            ("call", 0),   # P2 calls raise
        ])
        
        # This lambda consumes the iterator for every player
        t.bot.action = lambda *args: next(moves)
        
        # Start betting (starts at P1 because P0 is Button)
        t.betting_round(hands, board, active, contrib, 0)
        
        # Assertions
        self.assertEqual(contrib, [100, 100, 100])
        
        # Verify iterator is exhausted (proof that exactly those moves happened)
        self.assertRaises(StopIteration, next, moves)

    def test_betting_termination_all_in(self):
        """
        Tests that the loop terminates correctly when a player goes All-In 
        for LESS than the current bet.
        """
        t = pe.Table(n=2, stack=1000)
        t.stacks[1] = 50 # P1 is short stacked
        
        hands = [[(14,'s'),(14,'h')]]*2
        active = [True, True]
        contrib = [0, 0]
        
        # Scripted Sequence:
        # P1: Check
        # P0: Raise 100
        # P1: Call (can only put in 50, goes All-in)
        # Round should end immediately.
        
        moves = iter([
            ("check", 0),   # P1
            ("raise", 100), # P0
            ("call", 0)     # P1 (All-in logic handles the amount cap)
        ])
        
        t.bot.action = lambda *args: next(moves)
        
        t.betting_round(hands, [], active, contrib, 0)
        
        self.assertEqual(t.stacks[1], 0)   # P1 empty
        self.assertEqual(contrib[1], 50)   # P1 capped at 50
        self.assertEqual(contrib[0], 100)  # P0 full 100
        self.assertTrue(True, "Loop terminated successfully")

    def test_betting_fold_logic(self):
        """
        Tests that folding correctly removes a player from active status 
        and terminates the round if only 1 remains.
        """
        t = pe.Table(n=3, stack=1000)
        hands = [[(2,'s'),(3,'s')]] * 3
        active = [True, True, True]
        contrib = [0, 0, 0]
        
        # Script: P1 Check, P2 Fold, P0 Fold. 
        # P1 wins immediately (round returns).
        moves = iter([
            ("check", 0),
            ("fold", 0),
            ("fold", 0)
        ])
        
        t.bot.action = lambda *args: next(moves)
        
        t.betting_round(hands, [], active, contrib, 0)
        
        self.assertEqual(active, [False, True, False]) # Only P1 active
        # P1 should not have been asked to act again
        self.assertRaises(StopIteration, next, moves)

if __name__ == '__main__':
    unittest.main()