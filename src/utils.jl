###########################################################################
#                            Helper Functions                             #
###########################################################################

# This finds the root of the tree from any node in the tree
function find_root(node::BB.AbstractNode)::BB.AbstractNode
    pnode = node
    while !isnothing(pnode.parent)
        pnode = pnode.parent
    end
    return pnode
end
