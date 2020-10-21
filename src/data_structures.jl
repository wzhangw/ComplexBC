###########################################################################
#                             Data Structures                             #
###########################################################################

# A new type specifically for spatial branch and bound, essentially SDPWRMPowerModel
mutable struct NodeWRMPowerModel <: PM.AbstractWRMModel PM.@pm_fields end

# A new branch type, mapping (Wij, Tij) to their lower and upper bounds
mutable struct ComplexVariableBranch <: BB.AbstractBranch
    lb::Dict{Tuple{JuMP.VariableRef, JuMP.VariableRef},Real} # Lij
    ub::Dict{Tuple{JuMP.VariableRef, JuMP.VariableRef},Real} # Uij
end

# This computes coefficients for cuts (3a), (3b) in the paper
function compute_π(bounds::NTuple{6, <:Real})::NTuple{5, <:Real}
    function sigmoid(x::Real)::Real
        x == 0 ? res = 0 : res = (sqrt(1 + x^2) - 1) / x
        return res
    end    
    (Lii, Uii, Ljj, Ujj, Lij, Uij) = bounds
    π0 = -sqrt(Lii * Ljj * Uii * Ujj)
    π1 = -sqrt(Ljj * Ujj)
    π2 = -sqrt(Lii * Uii)
    π3 = (sqrt(Lii) + sqrt(Uii)) * (sqrt(Ljj) + sqrt(Ujj)) * (1 - sigmoid(Lij) * sigmoid(Uij)) / (1 + sigmoid(Lij) * sigmoid(Uij))
    π4 = (sqrt(Lii) + sqrt(Uii)) * (sqrt(Ljj) + sqrt(Ujj)) * (sigmoid(Lij) + sigmoid(Uij)) / (1 + sigmoid(Lij) * sigmoid(Uij))
    return (π0, π1, π2, π3, π4)
end

# This is the main branch type used in our implementation
# (i,j) should always be one of the lines in the power system
mutable struct SpatialBCBranch <: BB.AbstractBranch
    i::Int
    j::Int
    wii::JuMP.VariableRef
    wjj::JuMP.VariableRef
    wr::JuMP.VariableRef
    wi::JuMP.VariableRef
    mod_branch::Union{Nothing, ComplexVariableBranch, BB.VariableBranch} # this captures either a changed bound for Wij and Tij, or a changed bound for Wii/Wjj 
    bounds::NTuple{6, <:Real} # Lii, Uii, Ljj, Ujj, Lij, Uij
    valid_ineq_coeffs::NTuple{5, <:Real} # π coefficients
end

# create a SpatialBCBranch, in which mod_branch is nothing (because the exact matrix entry to branch is not decided yet)
function create_sbc_branch(i::Int64, j::Int64, prev_node::BB.AbstractNode)
    root = find_root(prev_node)
    pm = root.auxiliary_data["PM"]
    bus_ids = PM.ids(pm, :bus)
    lookup_w_index = Dict((bi,i) for (i,bi) in enumerate(bus_ids))
    widx_i = lookup_w_index[i]
    widx_j = lookup_w_index[j]
    wii = PM.var(pm, :WR)[widx_i,widx_i]
    wjj = PM.var(pm, :WR)[widx_j,widx_j]
    wr = PM.var(pm, :WR)[widx_i,widx_j]
    wi = PM.var(pm, :WI)[widx_i,widx_j]
    bounds = get_LU_from_branches(prev_node, i, j)
    πs = compute_π(bounds)
    return SpatialBCBranch(i, j, wii, wjj, wr, wi, nothing, bounds, πs)
end

# This function tracks the branches and update the bounds for selected entries (i,j)
function get_LU_from_branches(node::BB.AbstractNode, i::Int64, j::Int64)::NTuple{6, <:Real}
    pnode = node
    LU = [NaN for i in 1:6]
    while !isnothing(pnode.parent)
        sbc_branch = pnode.branch
        mod_branch = sbc_branch.mod_branch
        if sbc_branch.i == i && sbc_branch.j == j
            for i in eachindex(LU)
                if isnan(LU[i]) LU[i] = sbc_branch.bounds[i] end
            end
        elseif sbc_branch.i == i
            if isnan(LU[1]) LU[1] = sbc_branch.bounds[1] end
            if isnan(LU[2]) LU[2] = sbc_branch.bounds[2] end
        elseif sbc_branch.i == j
            if isnan(LU[3]) LU[3] = sbc_branch.bounds[1] end
            if isnan(LU[4]) LU[4] = sbc_branch.bounds[2] end
        elseif sbc_branch.j == i
            if isnan(LU[1]) LU[1] = sbc_branch.bounds[3] end
            if isnan(LU[2]) LU[2] = sbc_branch.bounds[4] end
        elseif sbc_branch.j == j
            if isnan(LU[3]) LU[3] = sbc_branch.bounds[3] end
            if isnan(LU[4]) LU[4] = sbc_branch.bounds[4] end
        end
        pnode = pnode.parent
    end
    if isnan(LU[1]) LU[1] = pnode.auxiliary_data["Lii"][i] end
    if isnan(LU[2]) LU[2] = pnode.auxiliary_data["Uii"][i] end
    if isnan(LU[3]) LU[3] = pnode.auxiliary_data["Lii"][j] end
    if isnan(LU[4]) LU[4] = pnode.auxiliary_data["Uii"][j] end
    if isnan(LU[5]) LU[5] = pnode.auxiliary_data["Lij"][(i,j)] end
    if isnan(LU[6]) LU[6] = pnode.auxiliary_data["Uij"][(i,j)] end

    return Tuple(LU)
end
