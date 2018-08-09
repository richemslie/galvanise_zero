from ggplib.util.symbols import SymbolFactory, ListTerm, Term

from ggpzero.defs.datadesc import Sample


def reflect_vertical(x, y, x_cords, y_cords):
    x_idx = x_cords.index(x)
    return x_cords[len(x_cords) - x_idx - 1], y


def reflect_horizontal(x, y, x_cords, y_cords):
    y_idx = y_cords.index(y)
    return x, y_cords[len(y_cords) - y_idx - 1]


def rotate_90(x, y, x_cords, y_cords):
    ' clockwise '
    assert len(x_cords) == len(y_cords)
    x_idx = x_cords.index(x)
    y_idx = y_cords.index(y)

    return x_cords[y_idx], y_cords[len(x_cords) - x_idx - 1]


symbol_factory = None


def symbolize(txt, pos):
    global symbol_factory
    if symbol_factory is None:
        symbol_factory = SymbolFactory()

    symbols = symbol_factory.symbolize(txt)
    symbols = symbols[pos]

    # convert terms to lists - make things simpler knowing it is a list of size 1+
    if isinstance(symbols, Term):
        return ListTerm((symbols,))

    return symbols


class Translater(object):
    def __init__(self, game_info, x_cords, y_cords):
        self.game_info = game_info
        self.x_cords = x_cords
        self.y_cords = y_cords

        self.base_symbols = [symbolize(b, 1) for b in self.game_info.model.bases]

        # the indexes into base for x/y
        self.base_root_term_indexes = {}

        # map base root term -> dict
        #      * maps from non-root terms -> index into model.bases
        self.base_root_term_to_mapping = {}
        self.skip_base_root_terms = set()

        self.action_list = []
        for ri in range(len(self.game_info.model.roles)):
            self.action_list.append([symbolize(a, 2) for a in self.game_info.model.actions[ri]])

        self.action_root_term_indexes = {}
        self.skip_action_root_term = set()

    def add_basetype(self, root_term, x_term_idx, y_term_idx):
        # create a dict for this root_term.  the dict will a be a mapping from x, y -> position on model
        assert root_term not in self.base_root_term_indexes
        self.base_root_term_indexes[root_term] = x_term_idx, y_term_idx

        mapping = self.base_root_term_to_mapping[root_term] = {}

        for model_bases_indx, terms in enumerate(self.base_symbols):
            if terms[0] != root_term:
                continue

            mapping[terms[1:]] = model_bases_indx

        print mapping

    def add_action_type(self, root_term, x_term_idx, y_term_idx):
        self.action_root_term_indexes[root_term] = x_term_idx, y_term_idx

    def add_skip_base(self, root_term):
        self.skip_base_root_terms.add(root_term)

    def add_skip_action(self, root_term):
        self.skip_action_root_term.add(root_term)

    def translate_terms(self, terms, x_term_idx, y_term_idx, do_reflection, rot_count):
        x, y = terms[x_term_idx], terms[y_term_idx]

        if do_reflection:
            x, y = reflect_vertical(x, y, self.x_cords, self.y_cords)

        for _ in range(rot_count):
            x, y = rotate_90(x, y, self.x_cords, self.y_cords)

        new_terms = list(terms)
        new_terms[x_term_idx] = Term(x)
        new_terms[y_term_idx] = Term(y)
        return ListTerm(new_terms)

    def translate_basestate(self, basestate, do_reflection, rot_count):
        # takes tuple/list, return new list (including self)
        assert isinstance(basestate, (tuple, list))
        assert len(basestate) == len(self.base_symbols)

        new_basestate = [0 for _ in range(len(basestate))]
        for indx, (value, terms) in enumerate(zip(basestate, self.base_symbols)):
            if not value:
                continue

            if terms[0] in self.skip_base_root_terms:
                new_basestate[indx] = 1
                continue

            if terms[0] not in self.base_root_term_indexes:
                raise Exception("Not supported base %s" % str(terms))

            x_term_idx, y_term_idx = self.base_root_term_indexes[terms[0]]
            new_terms = self.translate_terms(terms, x_term_idx, y_term_idx, do_reflection, rot_count)

            # set value on new basestate
            base_term, extra_terms = new_terms[0], new_terms[1:]
            new_bs_indx = self.base_root_term_to_mapping[base_term][extra_terms]
            new_basestate[new_bs_indx] = 1

        return new_basestate

    def translate_action(self, role_index, legal, do_reflection, rot_count):
        terms = self.action_list[role_index][legal]

        root_term = terms[0]
        if root_term in self.skip_action_root_term:
            return legal

        # convert the action
        x_term_idx, y_term_idx = self.action_root_term_indexes[terms[0]]
        new_terms = self.translate_terms(terms, x_term_idx, y_term_idx, do_reflection, rot_count)

        print terms, new_terms
        for legal_idx, other in enumerate(self.action_list[role_index]):
            if new_terms == other:
                return legal_idx

        assert False, "Did not find translation"

    def translate_policies(self, policies, do_reflection, rot_count):
        new_policies = []
        for role_index, policy in enumerate(policies):
            role_policy = []

            # get the action from the model
            for legal, p in policy:
                translated_legal = self.translate_action(role_index, legal, do_reflection, rot_count)
                role_policy.append(translated_legal, p)

            new_policies.append(role_policy)


def augment_samples(sample, game_info, game_desc, game_symmetries):
    # create the translator
    t = Translater(game_info, game_desc.x_cords, game_desc.y_cords)
    for ab in game_symmetries.apply_bases:
        t.add_basetype(ab.base_term, ab.x_term_idx, ab.y_term_idx)

    for ac in game_symmetries.apply_actions:
        t.add_action_type(ac.base_term, ac.x_term_idx, ac.y_term_idx)

    for term in game_symmetries.skip_bases:
        t.skip_bases(term)

    for term in game_symmetries.skip_actions:
        t.skip_actions(term)

    # define a prescription of what rotation/reflections to do
    if game_symmetries.do_rotations:
        prescription = [(False, 1), (False, 2), (False, 3)]

        if game_symmetries.do_reflection:
            prescription += [(True, 0), (True, 1), (True, 2), (True, 3)]
    else:
        assert game_symmetries.do_reflection
        prescription = [(True, 0)]

    for do_reflection, rot_count in prescription:

        # translate states/policies
        state = t.translate_basestate(tuple(sample.state), do_reflection, rot_count)
        prev_states = [t.translate_basestate(tuple(s), do_reflection, rot_count) for s in sample.prev_states]
        policies = t.translate_policies(sample.policies)

        match_identifier = sample.match_identifier + "_+%d_+%d" % (do_reflection, rot_count)

        yield Sample(state=state,
                     prev_states=prev_states,
                     policies=policies,
                     match_identifier=match_identifier,

                     # rest the same as sample
                     final_score=sample.final_score[:],
                     depth=sample.depth,
                     game_length=sample.game_length,
                     has_resigned=sample.has_resigned,
                     resign_false_positive=sample.resign_false_positive,
                     starting_sample_depth=sample.starting_sample_depth,
                     resultant_puct_score=sample.resultant_puct_score[:],
                     resultant_puct_visits=sample.resultant_puct_visits)
