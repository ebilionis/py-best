"""The base class of all binary trees.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""


__all__ = ['BinaryTree']


class BinaryTree(object):
    """The base class of all binary trees."""

    # Parent node
    _parent = None

    # Left child
    _left = None

    # Right child
    _right = None

    @property
    def parent(self):
        """Get parent element."""
        return self._parent

    @parent.setter
    def parent(self, value):
        """Set the parent element."""
        if not isinstance(value, BinaryTree):
            raise TypeError('Parent element must be a BinaryTree.')
        self._parent = value

    @property
    def left(self):
        """Get the left child."""
        return self._left

    @left.setter
    def left(self, value):
        """Set the left child."""
        if not isinstance(value, BinaryTree):
            raise TypeError('Left element must be a BinaryTree.')
        value.parent = self
        self._left = value

    @property
    def right(self):
        """Get the right child."""
        return self._right

    @right.setter
    def right(self, value):
        """Set the right child."""
        if not isinstance(value, BinaryTree):
            raise TypeError('Right element must be a BinaryTree.')
        value.parent = self
        self._right = value

    @property
    def has_left(self):
        """Return True if we have a left child."""
        return self.left is not None

    @property
    def has_right(self):
        """Return True if we have a right child."""
        return self.right is not None

    @property
    def has_parent(self):
        """Return True if we have a parent."""
        return self.parent is not None

    @property
    def is_root(self):
        """Return True if we are the root of the tree."""
        return not self.has_parent

    @property
    def is_left(self):
        """Return True if it is a left child."""
        return self.has_parent and self.parent.left is self

    @property
    def is_right(self):
        """Return True if it is a right child."""
        return self.has_parent and self.parent.right is self

    @property
    def is_leaf(self):
        """Return true if the element is a leaf of the tree."""
        return not (self.has_left and self.has_right)

    @property
    def num_leaves(self):
        """Get the number of elements."""
        if self.is_leaf:
            return 1
        r = 0
        if self.has_left:
            r += self.left.num_leaves
        if self.has_right:
            r += self.right.num_leaves
        return r

    @property
    def total_num_leaves(self):
        """Get the total number of leaves."""
        r = 1
        if self.has_left:
            r += self.left.total_num_leaves
        if self.has_right:
            r += self.right.total_num_leaves
        return r

    @property
    def leaves(self):
        """Get a list of a the leaves of the tree."""
        if self.is_leaf:
            return [self]
        s = []
        if self.has_left:
            s += self.left.leaves
        if self.has_right:
            s += self.right.leaves
        return s


    def __init__(self):
        """Initialize the object.

        Keyword Arguments:
            parent  ---     The parent of the node that is created.

        """
        pass
