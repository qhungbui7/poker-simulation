import random
import itertools
from collections import Counter

# --- Configuration & Constants ---
RANKS = list(range(2, 15))
SUITS = "cdhs"
HUMAN_ID = 0
MODE = "learn"  # options: learn, play, eval
BET_SIZES = [0.5, 1.0] # Pot sized bets
N_MC = 500

# --- Card Utilities ---
def format_card(card):
    r, s = card
    m = {11: "J", 12: "Q", 13: "K", 14: "A"}
    sm = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
    return f"{m.get(r, r)}{sm.get(s, s)}"

def make_deck():
    return [(r, s) for r in RANKS for s in SUITS]

def deal(deck, n):
    return [deck.pop() for _ in range(n)]

# --- C. Hand Evaluator (Pure, Deterministic, Total Ordering) ---
def best_hand_7(cards):
    """
    Evaluates 7 cards and returns a comparable tuple (category_score, tie_breakers).
    Lower index in tuple is more significant.
    """
    def ranks(c): return [r for r, _ in c]
    
    def eval5(c):
        r_list = ranks(c)
        rc = Counter(r_list)
        sc = Counter(s for _, s in c)
        uniq_desc = sorted(set(r_list), reverse=True)
        
        def find_straight_high(sorted_desc):
            # Normal straight
            r = sorted_desc[:]
            # Wheel check (A-2-3-4-5)
            if 14 in r: r.append(1)
            for i in range(len(r) - 4):
                if r[i] - r[i + 4] == 4:
                    return r[i]
            return None

        flush_suit = next((s for s, cnt in sc.items() if cnt >= 5), None)
        
        # 8. Straight Flush
        if flush_suit:
            flush_ranks = sorted(set(r for r, s in c if s == flush_suit), reverse=True)
            sf_high = find_straight_high(flush_ranks)
            if sf_high: return (8, (sf_high,))

        straight_high = find_straight_high(uniq_desc)
        counts = sorted(rc.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # 7. Four of a Kind
        if counts[0][1] == 4:
            return (7, (counts[0][0], max(r for r in uniq_desc if r != counts[0][0])))
            
        # 6. Full House
        if counts[0][1] == 3 and len(counts) > 1 and counts[1][1] >= 2:
            return (6, (counts[0][0], counts[1][0]))
        
        # 5. Flush
        if flush_suit:
            # top 5 cards of the flush suit
            return (5, tuple(sorted([r for r, s in c if s == flush_suit], reverse=True)[:5]))
        
        # 4. Straight
        if straight_high:
            return (4, (straight_high,))
        
        # 3. Trips
        if counts[0][1] == 3:
            k = tuple([r for r in uniq_desc if r != counts[0][0]][:2])
            return (3, (counts[0][0],) + k)
        
        # 2. Two Pair
        if len(counts) >= 2 and counts[0][1] == 2 and counts[1][1] == 2:
            k = next((r for r in uniq_desc if r not in (counts[0][0], counts[1][0])), 0)
            return (2, (counts[0][0], counts[1][0], k))
        
        # 1. Pair
        if counts[0][1] == 2:
            k = tuple([r for r in uniq_desc if r != counts[0][0]][:3])
            return (1, (counts[0][0],) + k)
        
        # 0. High Card
        return (0, tuple(uniq_desc[:5]))

    best = (-1, ())
    for c in itertools.combinations(cards, 5):
        v = eval5(c)
        if v > best: best = v
    return best

# --- Stats & AI Stub ---
class Stats:
    def __init__(self):
        self.hands = 0
        self.vpip = 0
        self.pfr = 0
    def vpip_rate(self): return self.vpip / self.hands if self.hands else 0

class AdaptiveBot:
    def __init__(self, stats): self.stats = stats
    def action(self, hole, board, need, pot, stack, valid_actions):
        # Extremely simple baseline
        if "check" in valid_actions and need == 0: return "check", 0
        if "call" in valid_actions and need > 0: return "call", 0
        return "fold", 0

# --- A. Side Pot Construction (Core Requirement) ---
def build_side_pots(contrib, active):
    """
    Constructs main and side pots based on player contributions.
    Returns a list of tuples: (pot_amount, list_of_eligible_player_indices)
    
    Algorithm:
    1. Identify all unique positive contribution levels from *active* players.
    2. Sort levels ascending.
    3. Slice total contributions (including folded players) into these levels.
    """
    # Active players with chips in play define the "caps"
    active_contribs = sorted(list(set(c for i, c in enumerate(contrib) if active[i] and c > 0)))
    
    pots = []
    last_level = 0
    
    for level in active_contribs:
        chunk_size = level - last_level
        pot_amount = 0
        eligible = []
        
        # Calculate who pays into this chunk and who is eligible
        for i in range(len(contrib)):
            # How much does player i contribute to this specific slice?
            # They contribute max(0, min(total_contrib, level) - last_level)
            payment = max(0, min(contrib[i], level) - last_level)
            pot_amount += payment
            
            # If active and put in at least this level (implied by active logic), they are eligible
            # We explicitly check strictly: active AND full contribution >= level
            if active[i] and contrib[i] >= level:
                eligible.append(i)
        
        if pot_amount > 0:
            pots.append({'amount': pot_amount, 'eligible': eligible})
        
        last_level = level
        
    return pots

# --- B. Betting State Machine ---
class Table:
    def __init__(self, n=6, stack=1000, sb=5, bb=10):
        self.n = n
        self.stacks = [stack] * n
        self.btn = 0
        self.sb = sb
        self.bb = bb
        self.stats = [Stats() for _ in range(n)]
        self.bot = AdaptiveBot(self.stats)

    def get_valid_actions(self, i, current_max, my_contrib, my_stack):
        actions = ["fold"]
        to_call = current_max - my_contrib
        
        if to_call == 0:
            actions.append("check")
            if my_stack > 0: actions.append("raise")
        elif to_call >= my_stack:
            actions.append("call") # effectively all-in
        else:
            actions.append("call")
            if my_stack > to_call: actions.append("raise")
            
        return actions

    def betting_round(self, hands, board, active, contrib, pot_start, is_preflop=False):
        """
        Executes a betting round using the provable termination conditions.
        """
        # State Initialization
        current_max_bet = max(contrib)
        last_raise_size = self.bb if is_preflop else 0
        
        # Who needs to act? Everyone active needs to check/call/fold unless all-in.
        # We track "acted_since_last_raise" flags.
        acted = [False] * self.n
        
        # Start pointer
        if is_preflop:
            # BB has acted implicitly (posted blind), but if raised, needs to act again.
            # However, in standard loops, we start UTG.
            start_i = (self.btn + 3) % self.n 
            # Pre-set blinds as not having fully acted relative to a potential raise?
            # Actually, standard machine: clear acted, loop starts.
            # Blinds are handled by initial contribution state.
            # Special case: BB is "live" if no raises.
            # To simplify: reset all acted to False.
            pass
        else:
            start_i = (self.btn + 1) % self.n

        idx = start_i
        
        while True:
            # TERMINATION CHECK:
            # Round ends if:
            # 1. Only 1 player active.
            # OR
            # 2. For ALL active players:
            #    (Player is All-In) OR (Player has Acted AND Player Contribution == Max Bet)
            
            active_count = sum(1 for x in active if x)
            if active_count <= 1:
                return # End immediately
            
            can_terminate = True
            for p in range(self.n):
                if not active[p]: continue
                if self.stacks[p] == 0: continue # All-in players are done
                
                # If player has stack, they must have acted AND matched the bet
                if not acted[p] or contrib[p] != current_max_bet:
                    can_terminate = False
                    break
            
            if can_terminate:
                return

            # Skip inactive or all-in players
            if not active[idx] or self.stacks[idx] == 0:
                idx = (idx + 1) % self.n
                continue

            # Player Action Required
            to_call = current_max_bet - contrib[idx]
            valid_acts = self.get_valid_actions(idx, current_max_bet, contrib[idx], self.stacks[idx])
            
            # --- Input / Bot Logic ---
            if idx == HUMAN_ID:
                print(f"\nYour turn. Pot: {pot_start + sum(contrib)}. To Call: {to_call}. Stack: {self.stacks[idx]}")
                print(f"Board: {' '.join(format_card(c) for c in board)}")
                print(f"Hand: {format_card(hands[idx][0])} {format_card(hands[idx][1])}")
                
                while True:
                    choice = input(f"Action {valid_acts}: ").strip().lower()
                    if choice == "r": choice = "raise"
                    if choice == "c": choice = "call"
                    if choice == "f": choice = "fold"
                    if choice == "k": choice = "check"
                    if choice in valid_acts: break
                
                amt = 0
                if choice == "raise":
                    min_r = max(last_raise_size, self.bb)
                    val = input(f"Raise amount (min {min_r}): ")
                    try:
                        amt = int(val)
                    except:
                        amt = min_r
                    if amt < min_r: amt = min_r
            else:
                # Simple bot logic
                choice, amt = self.bot.action(hands[idx], board, to_call, pot_start + sum(contrib), self.stacks[idx], valid_acts)

            # --- State Update ---
            if choice == "fold":
                active[idx] = False
                acted[idx] = True # Irrelevant now, but correct state
            
            elif choice == "check":
                acted[idx] = True
            
            elif choice == "call":
                amount = min(self.stacks[idx], to_call)
                self.stacks[idx] -= amount
                contrib[idx] += amount
                acted[idx] = True
                
            elif choice == "raise":
                # Raise logic: Call the current bet + Add raise amount
                call_part = current_max_bet - contrib[idx]
                total_needed = call_part + amt
                
                actual_pay = min(self.stacks[idx], total_needed)
                self.stacks[idx] -= actual_pay
                contrib[idx] += actual_pay
                
                new_raise_size = contrib[idx] - current_max_bet
                
                # If the raise was substantial (>= min raise), it re-opens action
                # If all-in for less than min-raise, it usually does NOT re-open action for those who already acted (not implemented here for simplicity, assuming all raises reopen)
                if new_raise_size > 0:
                    current_max_bet = contrib[idx]
                    last_raise_size = new_raise_size
                    # Reset acted flags for everyone else who is active
                    for i in range(self.n):
                        if i != idx and active[i] and self.stacks[i] > 0:
                            acted[i] = False
                
                acted[idx] = True

            idx = (idx + 1) % self.n

    def next_hand(self):
        # 1. Setup
        deck = make_deck()
        random.shuffle(deck)
        hands = [deal(deck, 2) for _ in range(self.n)]
        board = []
        active = [True] * self.n
        
        # SB/BB Posting
        sb_pos = (self.btn + 1) % self.n
        bb_pos = (self.btn + 2) % self.n
        
        # Contrib array tracks chips committed *in the current street*
        # We merge into 'pot' at end of street.
        contrib = [0] * self.n
        pot = 0
        
        # Post Blinds
        sb_amt = min(self.stacks[sb_pos], self.sb)
        self.stacks[sb_pos] -= sb_amt
        contrib[sb_pos] += sb_amt
        
        bb_amt = min(self.stacks[bb_pos], self.bb)
        self.stacks[bb_pos] -= bb_amt
        contrib[bb_pos] += bb_amt
        
        # 2. Preflop
        self.betting_round(hands, board, active, contrib, pot, is_preflop=True)
        
        # Merge Contribs
        pot += sum(contrib)
        
        # Uncalled Bet Return Logic (Simplified)
        # If one player bet way more than everyone else, the excess is technically returned.
        # For this engine, we rely on Side Pot logic to handle distribution, 
        # but pure poker rules return the excess immediately.
        # We will keep it simple: Side pot 1 with 1 player is returned.
        
        contrib_history = list(contrib) # Store for records if needed
        contrib = [0] * self.n # Reset for next street

        # 3. Flop
        if sum(active) > 1:
            board += deal(deck, 3)
            self.betting_round(hands, board, active, contrib, pot)
            pot += sum(contrib)
            for i, c in enumerate(contrib): contrib_history[i] += c
            contrib = [0] * self.n

        # 4. Turn
        if sum(active) > 1:
            board += deal(deck, 1)
            self.betting_round(hands, board, active, contrib, pot)
            pot += sum(contrib)
            for i, c in enumerate(contrib): contrib_history[i] += c
            contrib = [0] * self.n

        # 5. River
        if sum(active) > 1:
            board += deal(deck, 1)
            self.betting_round(hands, board, active, contrib, pot)
            pot += sum(contrib)
            for i, c in enumerate(contrib): contrib_history[i] += c
        
        # 6. Showdown & Distribution
        print("\n=== Showdown ===")
        print(f"Board: {' '.join(format_card(c) for c in board)}")
        
        # Use total contributions across all streets for side pot calc
        pots = build_side_pots(contrib_history, active)
        
        for p_idx, pot_obj in enumerate(pots):
            amt = pot_obj['amount']
            eligible = pot_obj['eligible']
            
            if not eligible: 
                continue # Should not happen if logic is correct
                
            # Evaluate hands for eligible players
            best_val = (-1,)
            winners = []
            
            # Find winner(s) for this specific side pot
            evals = []
            for pid in eligible:
                score = best_hand_7(hands[pid] + board)
                evals.append((pid, score))
                if score > best_val:
                    best_val = score
                    winners = [pid]
                elif score == best_val:
                    winners.append(pid)
            
            # Distribute
            if winners:
                share = amt // len(winners)
                rem = amt % len(winners)
                print(f"Pot {p_idx+1} ({amt}): Winners {winners} (Hand Class {best_val[0]})")
                for w in winners:
                    self.stacks[w] += share
                # Odd chips to first position (simplified)
                if rem: self.stacks[winners[0]] += rem
        
        self.btn = (self.btn + 1) % self.n
        print(f"Stacks: {self.stacks}")

if __name__ == "__main__":
    t = Table()
    t.next_hand()