###########################################################################
#                        Key Steps Implementation                         #
###########################################################################

# This function finds the line index (i,j) that the corresponding 2-by-2 W matrix attains the maximum λmin
function find_min_eigen(node::BB.AbstractNode)::Tuple{Int64, Int64}
    root = find_root(node)
    pm = root.auxiliary_data["PM"]
    bus_ids = PM.ids(pm, :bus)
    lookup_w_index = Dict((bi,i) for (i,bi) in enumerate(bus_ids))
    wr = PM.var(pm, :WR)
    wi = PM.var(pm, :WI)
    max_lambda = -Inf
    max_id = ()
    node_solution = node.solution
    eigenvalues = Dict()
    # for raw_k in bus_ids, raw_l in bus_ids
    #     k = lookup_w_index[raw_k]
    #     l = lookup_w_index[raw_l]
    #     if k < l
    #         lambda = 0.5 * (node_solution[wr[k,k]] + node_solution[wr[l,l]] - norm( [node_solution[wr[k,k]] - node_solution[wr[l,l]], 2 * node_solution[wr[k,l]], 2 * node_solution[wi[k,l]]] ) )
    #         println("Smallest eigenvalue for index pair ($(k), $(l)): ", lambda)
    #     end
    # end
    for (raw_i,raw_j) in PM.keys(root.auxiliary_data["Cuts"]["angle_lb"])
        i = lookup_w_index[raw_i]
        j = lookup_w_index[raw_j]
        lambda = 0.5 * (node_solution[wr[i,i]] + node_solution[wr[j,j]] - norm( [node_solution[wr[i,i]] - node_solution[wr[j,j]], 2 * node_solution[wr[i,j]], 2 * node_solution[wi[i,j]]] ) )
        eigenvalues[(raw_i, raw_j)] = lambda
        # println(lambda)
        if lambda > max_lambda
            max_id = (raw_i,raw_j)
            max_lambda = lambda
        end
    end
    node.auxiliary_data["eigenvalues"] = eigenvalues
    # println()
    # println(max_lambda)
    return max_id
end

# this is the interface for finding the complex entry to branch
function find_branching_entry(branch::SpatialBCBranch, node::BB.AbstractNode)::Tuple{Int64, Int64}
    root = find_root(node)
    model = root.model
    opt_idx = solve_candidate_nodes(branch, node)
    return opt_idx
end

# this implements weak branching to find the best, and returns the index of the complex entry to branch on
function solve_candidate_nodes(branch::SpatialBCBranch, node::BB.AbstractNode)
    function _make_fake_branch(branch::SpatialBCBranch, mode::String)
        (Lii, Uii, Ljj, Ujj, Lij, Uij) = branch.bounds
        if mode == "Wii_up" Lii = (Lii + Uii) / 2
        elseif mode == "Wii_down" Uii = (Lii + Uii) / 2
        elseif mode == "Wjj_up" Ljj = (Ljj + Ujj) / 2
        elseif mode == "Wjj_down" Ujj = (Ljj + Ujj) / 2
        elseif mode == "Wij_up" Lij = (Lij + Uij) / 2
        elseif mode == "Wij_down" Uij = (Lij + Uij) / 2
        end
        new_bounds = (Lii, Uii, Ljj, Ujj, Lij, Uij)
        new_π = compute_π(new_bounds)
        return SpatialBCBranch(branch.i, branch.j, branch.wii, branch.wjj, branch.wr, branch.wi, nothing, 
                                new_bounds, new_π )
    end
    
    function _add_constraints_from_fake_branch!(model::JuMP.Model, branch::SpatialBCBranch)
        i = branch.i
        j = branch.j
        (Lii, Uii, Ljj, Ujj, Lij, Uij) = branch.bounds
        πs = branch.valid_ineq_coeffs
        wii = branch.wii
        wjj = branch.wjj
        wr = branch.wr
        wi = branch.wi
        new_branches = []
        JuMP.set_lower_bound(wii, Lii)
        JuMP.set_upper_bound(wii, Uii)
        JuMP.set_lower_bound(wjj, Ljj)
        JuMP.set_upper_bound(wjj, Ujj)
        JuMP.@constraint(model, Lij * wr <= wi)
        JuMP.@constraint(model, Uij * wr >= wi)
        JuMP.@constraint(model, πs[1] + πs[2] * wii + πs[3] * wjj + πs[4] * wr + πs[5] * wi >= Ujj * wii + Uii * wjj - Uii * Ujj)
        JuMP.@constraint(model, πs[1] + πs[2] * wii + πs[3] * wjj + πs[4] * wr + πs[5] * wi >= Ljj * wii + Lii * wjj - Lii * Ljj)
    end    
    i = branch.i
    j = branch.j
    root = find_root(node)
    modes = ["Wii", "Wjj", "Wij"]
    best_score = -Inf
    best_mode = ""
    for mode in modes
        fake_branch = _make_fake_branch(branch, mode * "_up")
        model = JuMP.Model(MOSEK_OPTIMIZER)
        # model = JuMP.Model(SCS_OPTIMIZER)
        fake_branch.wii = JuMP.@variable(model, wii)
        fake_branch.wjj = JuMP.@variable(model, wjj)
        fake_branch.wr = JuMP.@variable(model, wr)
        fake_branch.wi = JuMP.@variable(model, wi)
        JuMP.@variable(model, λ)
        JuMP.@constraint(model, [wii + wjj - 2 * λ, wii - wjj, 2 * wr, 2 * wi] in JuMP.SecondOrderCone())
        JuMP.@objective(model, Max, λ)
        _add_constraints_from_fake_branch!(model, fake_branch)
        JuMP.optimize!(model)
        λ_up = JuMP.value(λ)

        fake_branch = _make_fake_branch(branch, mode * "_down")
        model = JuMP.Model(MOSEK_OPTIMIZER)
        fake_branch.wii = JuMP.@variable(model, wii)
        fake_branch.wjj = JuMP.@variable(model, wjj)
        fake_branch.wr = JuMP.@variable(model, wr)
        fake_branch.wi = JuMP.@variable(model, wi)
        JuMP.@variable(model, λ)
        JuMP.@constraint(model, [wii + wjj - 2 * λ, wii - wjj, 2 * wr, 2 * wi] in JuMP.SecondOrderCone())
        JuMP.@objective(model, Max, λ)
        _add_constraints_from_fake_branch!(model, fake_branch)
        JuMP.optimize!(model)
        λ_down = JuMP.value(λ)

        # compute score, update if better
        μ = 0.15
        score = μ * max(-λ_up, -λ_down) + (1-μ) * min(-λ_up, -λ_down)
        if score > best_score
            best_score = score
            best_mode = mode
        end
    end

    if best_mode == "Wii"
        return (i,i)
    elseif best_mode == "Wjj"
        return (j,j)
    else # "Wij"
        return (i,j)
    end
end

# this tracks all the branches above node and add constraints to model accordingly
# if there are multiple branches on the same complex entry, only add constraints for the one with largest depth (closest to node)
function backtracking!(model::JuMP.Model, node::BB.AbstractNode)
    node.auxiliary_data["prev_branch_crefs"] = JuMP.ConstraintRef[]
    root = find_root(node)
    pnode = node
    already_modified = []
    while !isnothing(pnode.parent)
        (i,j) = (pnode.branch.i, pnode.branch.j)
        if !((i,j) in already_modified)
            crefs = add_constraints_from_branch!(model, pnode.branch, root)
            for cref in crefs
                push!(node.auxiliary_data["prev_branch_crefs"], cref)
            end
            push!(already_modified, (i,j))
        end
        pnode = pnode.parent
    end
end

# this modifies constraints in model based on branch
function add_constraints_from_branch!(model::JuMP.Model, branch::SpatialBCBranch, root::BB.AbstractNode)::Vector{JuMP.ConstraintRef}
    i = branch.i
    j = branch.j
    (Lii, Uii, Ljj, Ujj, Lij, Uij) = branch.bounds
    πs = branch.valid_ineq_coeffs
    wii = branch.wii
    wjj = branch.wjj
    wr = branch.wr
    wi = branch.wi
    new_branches = []
    JuMP.set_lower_bound(wii, Lii)
    JuMP.set_upper_bound(wii, Uii)
    JuMP.set_lower_bound(wjj, Ljj)
    JuMP.set_upper_bound(wjj, Ujj)
    JuMP.set_normalized_coefficient(root.auxiliary_data["Cuts"]["angle_lb"][(i,j)], wr, Lij)
    JuMP.set_normalized_coefficient(root.auxiliary_data["Cuts"]["angle_ub"][(i,j)], wr, Uij)
    push!(new_branches, JuMP.@constraint(model, πs[1] + πs[2] * wii + πs[3] * wjj + πs[4] * wr + πs[5] * wi >= Ujj * wii + Uii * wjj - Uii * Ujj))
    push!(new_branches, JuMP.@constraint(model, πs[1] + πs[2] * wii + πs[3] * wjj + πs[4] * wr + πs[5] * wi >= Ljj * wii + Lii * wjj - Lii * Ljj))
    return new_branches
end

# this resets constraints (2) and deletes constraints (3) in model
function delete_prev_branch_constr!(model::JuMP.Model, node::BB.AbstractNode)
    root = find_root(node)
    pm = root.auxiliary_data["PM"]
    w = PM.var(pm, :w)
    wr = PM.var(pm, :wr)
    for (i,_) in PM.ref(pm, :bus)
        JuMP.set_lower_bound(w[i], root.auxiliary_data["Lii"][i])
        JuMP.set_upper_bound(w[i], root.auxiliary_data["Uii"][i])
    end
    for (pair,_) in PM.ref(pm, :buspairs)
        JuMP.set_normalized_coefficient(root.auxiliary_data["Cuts"]["angle_lb"][pair], wr[pair], root.auxiliary_data["Lij"][pair])
        JuMP.set_normalized_coefficient(root.auxiliary_data["Cuts"]["angle_ub"][pair], wr[pair], root.auxiliary_data["Uij"][pair])
    end
    while !isempty(node.auxiliary_data["prev_branch_crefs"])
        cref = Base.pop!(node.auxiliary_data["prev_branch_crefs"])
        JuMP.delete(model, cref)
    end
end
