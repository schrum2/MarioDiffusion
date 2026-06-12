
import heapq
import itertools


class GraphSearch:
    

    def __init__(self):
        self.visited = None
        self.expanded_by_key = None   # dominance buckets (see search)

    def search(self, start, reset=True, budget=None, continue_after_success=False, is_goal=None):
        """Return the list of actions from start to a goal, or None if unreachable

        start: initial state
        reset: clear the visited set before searching
        budget: max states expanded before giving up (raises RuntimeError); None means unlimited
        continue_after_success: if True, keep exploring after the goal is located
        is_goal: optional predicate(state)->bool overriding state.isGoal() (e.g. to search
                 for "any cell on an edge" rather than a single goal point)
        """
        goal_test = is_goal if is_goal is not None else (lambda s: s.isGoal())
        # Each fringe entry is (priority, tie_break, state, actions_so_far, cost_so_far).
        # The unique tie_break counter guarantees two entries never get compared by
        # state (states have no ordering), which would otherwise crash heapq on ties.
        counter = itertools.count()
        frontier = []
        self.push(frontier, counter, start, [], 0.0)

        if reset:
            self.visited = set()
            self.expanded_by_key = {}

        # States may opt into dominance pruning by defining dominance_key() and
        # dominates() (e.g. LodeRunnerState: same position, remaining gold a subset).
        # A popped state dominated by an already-expanded state is skipped, because
        # everything reachable from it is also reachable from the dominator. This
        # collapses state-space blowups like (position x remaining-gold subsets);
        # solutions stay valid, but are no longer guaranteed to be shortest.
        use_dominance = hasattr(start, "dominance_key") and hasattr(start, "dominates")
        if use_dominance and self.expanded_by_key is None:
            self.expanded_by_key = {}

        count = 0
        found = False
        solution = None

        while frontier and (not found or continue_after_success):
            _, _, state, actions, cost = heapq.heappop(frontier)

            if goal_test(state) and not found:
                solution = actions           # first goal popped is optimal (admissible h)
                found = True
            elif state not in self.visited:
                if use_dominance:
                    bucket = self.expanded_by_key.setdefault(state.dominance_key(), [])
                    if any(d.dominates(state) for d in bucket):
                        continue
                    # Dominance is transitive, so entries the new state dominates are
                    # redundant; dropping them keeps the buckets small.
                    bucket[:] = [d for d in bucket if not state.dominates(d)]
                    bucket.append(state)
                count += 1
                if budget is not None and count > budget:
                    raise RuntimeError(f"A* exceeded computation budget: {budget}")
                self.visited.add(state)      # closed set: never expand a state twice
                for next_state, action, step in state.get_successors():
                    if next_state is None:
                        continue
                    self.push(frontier, counter, next_state, actions + [action], cost + step)

        return solution

    def push(self, frontier, counter, state, actions, cost):
        priority = self.priority(state, cost)
        heapq.heappush(frontier, (priority, next(counter), state, actions, cost))

    def get_visited(self):
        return self.visited


class AStarSearch(GraphSearch):
    def __init__(self, heur):
        """heur : callable taking a state, returning estimated cost-to-goal

        For Mega Man pass MegaManState.orb_heuristic; it takes a state as its only
        argument, so it works directly as the callable here
        """
        super().__init__()
        self.heur = heur

    def priority(self, state, cost):
        return cost + self.heur(state)