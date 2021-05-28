import copy
from rule import Rule


class Pruner:

    def __init__(self, dataset, terms_mgr, sg_comparison):
        self._terms_mgr = terms_mgr
        self._dataset = dataset
        self.current_rule = None
        self._comparison = sg_comparison

    def prune(self, rule):
        """Prune rule's antecedent while its quality measure does not decrease
        or rule's length is greater than one condition.
        At each iteration all conditions are tested for removal, being chosen 
        the one which promotes maximum overall quality improvement.
        """
        print(10*'*', "THE PRUNER", 10*'*')
        print()
        self.current_rule = copy.deepcopy(rule)
        print("Received for pruning:", self.current_rule.antecedent)
        print()
        # r pode ser eliminado;
        # r salvar somente o antecedente da regra de entrada;
        # r obter antecedent.items como lista para iteração;
        # r criar Pruned rule aqui;

        if len(self.current_rule.antecedent) == 1:
            print("Cannot be pruned: one antecedent only.")
            print()

        pruning_iteration = 1
        while (len(self.current_rule.antecedent) > 1):

            pruned_rule_has_better_quality = False
            current_antecedent = self.current_rule.antecedent.copy()

            print("Pruning iteration {}:".format(pruning_iteration),
                  current_antecedent)

            for attr in current_antecedent:
                # r new pruned rule antecedent and cases
                pruned_rule = Rule(self._dataset, self._comparison)
                pruned_rule.antecedent = current_antecedent.copy()
                # r atribuir os antecedentes da regra criada fora do while
                pruned_rule.antecedent.pop(attr, None)
                pruned_rule.set_cases(
                    self._terms_mgr.get_cases(pruned_rule.antecedent)
                )
                pruned_rule.set_fitness()

                if pruned_rule.fitness >= self.current_rule.fitness:
                    pruned_rule_has_better_quality = True
                    self.current_rule = copy.deepcopy(pruned_rule)

            if not pruned_rule_has_better_quality:
                break

            pruning_iteration += 1

        print("Pruned rule:", self.current_rule.antecedent)
        print(self.current_rule.fitness)
        input()

        return self.current_rule
