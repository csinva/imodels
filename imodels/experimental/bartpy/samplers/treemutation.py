import numpy as np

from abc import abstractmethod, ABC
from typing import Optional, Tuple

from ..model import Model
from ..mutation import TreeMutation
from ..samplers.sampler import Sampler
from ..tree import Tree


class TreeMutationSampler(Sampler):
    """
    A sampler for tree mutation space.
    Responsible for producing samples of ways to mutate a tree within a model

    A general schema of implementation is to combine a proposer and likihood evaluator to:
     - propose a mutation
     - assess likihood
     - accept if likihood higher than a uniform(0, 1) draw
    """

    def sample(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        raise NotImplementedError()

    def step(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        raise NotImplementedError()


class TreeMutationProposer(ABC):
    """
    A TreeMutationProposer is responsible for generating samples from tree space
    It is capable of generating proposed TreeMutations
    """

    @abstractmethod
    def propose(self, tree: Tree) -> TreeMutation:
        """
        Propose a mutation to make to the given tree

        Parameters
        ----------
        tree: Tree
            The tree to be mutate

        Returns
        -------
        TreeMutation
            A way to update the input tree
        """
        raise NotImplementedError()


class TreeMutationLikihoodRatio(ABC):
    """
    Responsible for evaluating the ratio of mutations to the reverse movement
    """

    def log_probability_ratio(self, model: Model, tree: Tree, mutation: TreeMutation) -> Tuple[float, tuple, tuple]:
        """
        Calculated the ratio of the likihood of a mutation over the likihood of the reverse movement

        Main access point for the class

        Parameters
        ----------
        model: Model
            The overall model object the tree belongs to
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            The proposed mutation

        Returns
        -------
        float
            logged ratio of likelihoods
        """
        log_likelihood_ratio, (l_new, l_old) = self.log_likihood_ratio(model, tree, mutation)
        log_transition_ratio, (t_new, t_old) = self.log_transition_ratio(tree, mutation)
        log_prior_ratio, (p_new, p_old) = self.log_tree_ratio(model, tree, mutation)
        bayes_term = log_transition_ratio + log_prior_ratio
        prob_score = bayes_term + log_likelihood_ratio
        ratio = np.exp(prob_score)
        p_t_new = p_new + t_new
        p_t_old = p_old + t_old
        prob_new = l_new+p_t_new
        prob_old = l_old+p_t_old
        return ratio, (l_new, l_old), (prob_new, prob_old)

    @abstractmethod
    def log_transition_ratio(self, tree: Tree, mutation: TreeMutation) -> float:
        """
        The logged ratio of the likihood of making the transition to the likihood of making the reverse transition.
        e.g. in the case of using only grow and prune mutations:
            log(likihood of growing from tree to the post mutation tree / likihood of pruning from the post mutation tree to the tree)

        Parameters
        ----------
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            the proposed mutation

        Returns
        -------
        float
            logged likihood ratio
        """
        raise NotImplementedError()

    @abstractmethod
    def log_tree_ratio(self, model: Model, tree: Tree, mutation: TreeMutation) -> float:
        """
        Logged ratio of the likihood of the tree before and after the mutation
        i.e. the product of the probability of all split nodes being split and all leaf node note being split

        Parameters
        ----------
        model: Model
            The model the tree to be changed is part of
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            the proposed mutation

        Returns
        -------
        float
            logged likihood ratio
        """

        raise NotImplementedError()

    @abstractmethod
    def log_likihood_ratio(self, model: Model, tree: Tree, mutation: TreeMutation):
        """
        The logged ratio of the likihood of all the data points before and after the mutation
        Generally more complex trees should be able to fit the data better than simple trees

        Parameters
        ----------
        model: Model
            The model the tree to be changed is part of
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            the proposed mutation

        Returns
        -------
        float
            logged likihood ratio
        """
        raise NotImplementedError()
