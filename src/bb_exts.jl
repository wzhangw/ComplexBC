###########################################################################
#                 Extensions of BranchAndBound Functions                  #
#                   General Algorithm Flow Goes here                      #
###########################################################################

# return a "deepcopy" of SpatialBCBranch for everything except for variable references
# this is needed to ensure variable references in new branch copy are still valid
# the mod_branch will not be duplicated here if it is not nothing
function branch_copy(branch::SpatialBCBranch)
    return SpatialBCBranch(branch.i, branch.j, branch.wii, branch.wjj, branch.wr, branch.wi, 
                           branch.mod_branch, deepcopy(branch.bounds), deepcopy(branch.valid_ineq_coeffs))
end

# this uses trace minimization to test if the given incomplete wr, wi can recover a rank-1 complex matrix
# if yes, wr and wi will be updated with the complete rank-1 solution, and the function returns true
# if no, wr and wi will remain unchanged, and the function returns false
# TODO: modify this so that it takes into account cut lines if necessary
#=
function min_trace_solution!(wr::Matrix{Float64}, wi::Matrix{Float64}, pm::AbstractPowerModel)::Bool
    bus_ids = ids(pm, :bus)
    n_bus = length(bus_ids)
    lookup_w_index = Dict((bi, i) for (i, bi) in enumerate(bus_ids))

    @info "--- Eigenvalues of W solution from SDP relaxation: " svdvals(wr + im .* wi)[1:3]

    # set up trace minimization
    m = Model(MOSEK_OPTIMIZER)
    @variable(m, XR[1:n_bus, 1:n_bus])
    @variable(m, XI[1:n_bus, 1:n_bus])
    @constraint(m, [XR -XI; XI XR] in PSDCone())
    for i in bus_ids
        @constraint(m, XR[i,i] == wr[i,i])
    end
    for (i,j) in ids(pm, :buspairs)
        windex_i = lookup_w_index[i]
        windex_j = lookup_w_index[j]
        @constraint(m, XR[i,j] == wr[i,j])
        @constraint(m, XI[i,j] == wi[i,j])
        @constraint(m, XR[j,i] == wr[j,i])
        @constraint(m, XI[j,i] == wi[j,i])
    end
    # TODO: add codes here to take into account cut lines if necessary (for network decomposition later)
    @objective(m, Min, tr([XR -XI; XI XR]))
    optimize!(m)
    @info "--- Trace minimization termination status: $(JuMP.termination_status(m)) ---"

    # eigenvalue test
    sol = value.(XR) + im * value.(XI)
    # @info "--- Trace minimization solution: " sol
    rk = rank(sol, rtol = 1e-4)
    @info "--- Eigenvalues of W solution after trace minimization: " svdvals(sol)[1:3]
    @info "--- The rank of the trace minimization solution: $(rk) ---"
    if rk == 1
        wr = value.(XR)
        wi = value.(XI)
        return true
    end
    return false
end
=#
# simply check the w values returned by the solver, no special handling of free variables
function min_trace_solution!(wr::Matrix{Float64}, wi::Matrix{Float64}, pm::PM.AbstractPowerModel)::Bool
    solution = wr + im .* wi
    # @info "--- Eigenvalues: " svdvals(solution)
    return rank(solution, rtol = 1e-4) == 1
end

function BB.branch!(tree::BB.AbstractTree, node::BB.AbstractNode)
    @info " Node id $(node.id), status $(node.solution_status), bound $(node.bound)"
    root = find_root(node)
    model = root.model
    if node.bound >= tree.best_incumbent
        if isapprox(node.bound, tree.best_incumbent, rtol = 1e-5) && node.bound > root.auxiliary_data["best_bound"]
            root.auxiliary_data["best_bound"] = node.bound
            root.auxiliary_data["best_id"] = node.id
        end
        @info " Fathomed by bound"
    elseif node.depth >= 100
        if node.bound > root.auxiliary_data["best_bound"]
            root.auxiliary_data["best_bound"] = node.bound
            root.auxiliary_data["best_id"] = node.id
        end
        @info " Fathomed by maximum depth"
    elseif node.solution_status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.SLOW_PROGRESS, MOI.ALMOST_OPTIMAL]
        # determine the complex entry to branch on based on solution
        pm = root.auxiliary_data["PM"]
        (i,j) = find_min_eigen(node)
        WR = PM.var(pm, :WR)
        WI = PM.var(pm, :WI)
        wr = JuMP.value.(WR)
        wi = JuMP.value.(WI)
        is_rank1 = min_trace_solution!(wr, wi, root.auxiliary_data["PM"]) # this implements trace minimization for rank-1 check and recovery
        # eigvs = eigvals(w)
        # println(eigvs)
        # if node.auxiliary_data["eigenvalues"][(i,j)] <= 1e-8
        # if eigvs[end-1] <= 5e-5
        if is_rank1
            # update the node with best bound (including processed ones), recorded in root
            root.auxiliary_data["rank1"][node.id] = node.bound
            @info " Fathomed by reaching rank-1 solution"
        else
            new_sbc_branch = create_sbc_branch(i, j, node)
            (new_i,new_j) = find_branching_entry(new_sbc_branch, node)
    
            # create branches and child nodes accordingly
            up_bounds_arr = [k for k in new_sbc_branch.bounds]
            down_bounds_arr = [k for k in new_sbc_branch.bounds]
            bus_ids = PM.ids(pm, :bus)
            lookup_w_index = Dict((bi,i) for (i,bi) in enumerate(bus_ids))
            widx_new_i = lookup_w_index[new_i]
            widx_new_j = lookup_w_index[new_j]
            if new_i != new_j # branch on Wij
                up_bounds_arr[5] = (up_bounds_arr[5] + up_bounds_arr[6]) / 2
                down_bounds_arr[6] = (down_bounds_arr[5] + down_bounds_arr[6]) / 2
                vpair = (PM.var(pm, :WR)[widx_new_i,widx_new_j], PM.var(pm, :WI)[widx_new_i,widx_new_j])
                up_mod_branch = ComplexVariableBranch(Dict(vpair => up_bounds_arr[5]), Dict(vpair => up_bounds_arr[6]))
                down_mod_branch = ComplexVariableBranch(Dict(vpair => down_bounds_arr[5]), Dict(vpair => down_bounds_arr[6]))
                @info " Branch at W$(new_i)$(new_j), [L, U] breaks into by [$(down_bounds_arr[5]),$(up_bounds_arr[5]),$(up_bounds_arr[6])]."
            elseif new_i == i # branch on Wii
                up_bounds_arr[1] = (up_bounds_arr[1] + up_bounds_arr[2]) / 2
                down_bounds_arr[2] = (down_bounds_arr[1] + down_bounds_arr[2]) / 2
                v = PM.var(pm, :WR)[widx_new_i, widx_new_j]
                up_mod_branch = BB.VariableBranch(Dict(v => up_bounds_arr[1]), Dict(v => up_bounds_arr[2]))
                down_mod_branch = BB.VariableBranch(Dict(v => down_bounds_arr[1]), Dict(v => down_bounds_arr[2]))
                @info " Branch at W$(new_i)$(new_j), [L, U] breaks into by [$(down_bounds_arr[1]),$(up_bounds_arr[1]),$(up_bounds_arr[2])]."
            else # branch on Wjj
                up_bounds_arr[3] = (up_bounds_arr[3] + up_bounds_arr[4]) / 2
                down_bounds_arr[4] = (down_bounds_arr[3] + down_bounds_arr[4]) / 2
                v = PM.var(pm, :WR)[widx_new_i, widx_new_j]
                up_mod_branch = BB.VariableBranch(Dict(v => up_bounds_arr[3]), Dict(v => up_bounds_arr[4]))
                down_mod_branch = BB.VariableBranch(Dict(v => down_bounds_arr[3]), Dict(v => down_bounds_arr[4]))
                @info " Branch at W$(new_i)$(new_j), [L, U] breaks into by [$(down_bounds_arr[3]),$(up_bounds_arr[3]),$(up_bounds_arr[4])]."
            end
            next_branch_up = branch_copy(new_sbc_branch)
            next_branch_down = new_sbc_branch
            next_branch_up.bounds = Tuple(up_bounds_arr)
            next_branch_down.bounds = Tuple(down_bounds_arr)
            next_branch_up.mod_branch = up_mod_branch
            next_branch_down.mod_branch = down_mod_branch
            child_up = BB.create_child_node(node, next_branch_up)
            child_down = BB.create_child_node(node, next_branch_down)
            BB.push!(tree, child_up)
            BB.push!(tree, child_down)    
        end
    else
        @info " Fathomed by solution status: $(node.solution_status)"
    end
    push!(root.auxiliary_data["best_bounds"], root.auxiliary_data["best_bound"])
    delete_prev_branch_constr!(model, node)
end

# implement depth first rule
function BB.next_node(tree::BB.AbstractTree)
    # sort! is redefined to find the node with maximum depth (consistent with the paper's implementation)
    # sort!(tree::BB.AbstractTree) = Base.sort!(tree.nodes, by=x->x.depth)
    BB.sort!(tree)
    node = Base.pop!(tree.nodes)
    return node
end

function BB.termination(tree::BB.AbstractTree)
    @info "Tree nodes: processed $(length(tree.processed)), left $(length(tree.nodes)), total $(tree.node_counter), best bound $(tree.best_bound), best incumbent $(tree.best_incumbent)"
    if BB.isempty(tree)
        # @info "Completed the tree search"
        return true
    end
    if length(tree.processed) >= MAX_NODE_ALLOWED
        # @info "Reached node limit"
        return true
    end
    return false
end

# This makes original adjust_branch! invalid
BB.adjust_branch!(branch_objects::Array{SpatialBCBranch,1}) = nothing

# This makes original apply_changes! invalid
BB.apply_changes!(node::BB.JuMPNode) = nothing

function BB.bound!(node::BB.JuMPNode)
    model = find_root(node).model
    backtracking!(model, node)
    JuMP.optimize!(model)
    node.solution_status = JuMP.termination_status(model)
    if node.solution_status == MOI.INFEASIBLE || JuMP.dual_status(model) in [MOI.INFEASIBILITY_CERTIFICATE]
        node.bound = Inf
    elseif node.solution_status == MOI.DUAL_INFEASIBLE || JuMP.primal_status(model) in [MOI.INFEASIBILITY_CERTIFICATE]
        node.bound = -Inf
    elseif node.solution_status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.SLOW_PROGRESS, MOI.ALMOST_OPTIMAL]
        node.bound = JuMP.objective_value(model)
        vrefs = JuMP.all_variables(model)
        for v in vrefs
            node.solution[v] = JuMP.value(v)
        end
    else
        @warn "Unexpected node solution status: $(node.solution_status)"
        node.bound = -Inf
    end

    # store the solution in node
    solution = Dict{String, Float64}() # name => value
    for var in JuMP.all_variables(model)
        solution[JuMP.name(var)] = JuMP.value(var)
    end
    node.auxiliary_data["node solution"] = solution

end

# This finds an incumbent for each node, by converting SDP problem into rank-1 form and solving with Ipopt
#=
function BB.heuristics!(node::BB.JuMPNode)
    ipopt_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    root = find_root(node)
    tree = root.auxiliary_data["tree"]
    pm = root.auxiliary_data["PM"]
    exact_model = copy(root.model)
    buses = collect(keys(ref(pm, :bus)))
    JuMP.@variable(exact_model, vr[buses])
    JuMP.@variable(exact_model, vi[buses])

    # get rid of PSDCone constraint (for Ipopt)
    psdcone = JuMP.all_constraints(exact_model, Array{GenericAffExpr{Float64, VariableRef}, 1}, MOI.PositiveSemidefiniteConeSquare)[1]
    JuMP.delete(exact_model, psdcone)

    # add rank 1 constraints
    bus_ids = ids(pm, :bus)
    lookup_w_index = Dict((bi,i) for (i,bi) in enumerate(bus_ids))
    for i in buses, j in buses
        wi = lookup_w_index[i]
        wj = lookup_w_index[j]
        if i == j
            Wi = JuMP.variable_by_name(exact_model, "0_WR[$(i),$(i)]")
            Wj = JuMP.variable_by_name(exact_model, "0_WR[$(j),$(j)]")
            JuMP.@constraint(exact_model, Wi == vr[i]^2 + vi[i]^2)
            JuMP.@constraint(exact_model, Wj == vr[j]^2 + vi[j]^2)
        else
            if !isnothing(JuMP.variable_by_name(exact_model, "0_WR[$(i),$(j)]"))
                WR = JuMP.variable_by_name(exact_model, "0_WR[$(i),$(j)]")
            else
                WR = JuMP.variable_by_name(exact_model, "0_WR[$(j),$(i)]")
            end
            WI = JuMP.variable_by_name(exact_model, "0_WI[$(i),$(j)]")
            JuMP.@constraint(exact_model, WR == vr[i] * vr[j] + vi[i] * vi[j])
            JuMP.@constraint(exact_model, WI == vr[j] * vi[i] - vi[j] * vr[i])
        end
    end

    # convert second order cone constraints into quadratic constraints
    soc_crefs = JuMP.all_constraints(exact_model, Array{GenericAffExpr{Float64, VariableRef}, 1}, MOI.SecondOrderCone)
    for cref in soc_crefs
        cobj = JuMP.constraint_object(cref)
        terms = cobj.func
        JuMP.@constraint(exact_model, sum(terms[i]^2 for i in 2:length(terms)) <= terms[1]^2)
        JuMP.delete(exact_model, cref)
    end

    # convert rotated second order cone constraints into quadratic constraints (this is needed if objective is quadratic)
    rsoc_crefs = JuMP.all_constraints(exact_model, Array{GenericAffExpr{Float64, VariableRef}, 1}, MOI.RotatedSecondOrderCone)
    for cref in rsoc_crefs
        cobj = JuMP.constraint_object(cref)
        terms = cobj.func
        JuMP.@constraint(exact_model, sum(terms[i]^2 for i in 3:length(terms)) <= 2 * terms[1].constant * terms[2]) # hacky way to avoid higher order (nonlinear) terms
        JuMP.delete(exact_model, cref)
    end

    # initialize with sdp solution
    # sdp_sol = node.auxiliary_data["node solution"]
    # for (name, val) in sdp_sol
    #     var = JuMP.variable_by_name(exact_model, name)
    #     JuMP.set_start_value(var, val)
    # end

    # initialize with PowerModels default for ACRPowerModel
    JuMP.set_start_value.(vr, 1.0)

    JuMP.set_optimizer(exact_model, ipopt_optimizer)
    JuMP.optimize!(exact_model)
    @info " Heuristics result: $(objective_value(exact_model))"
    exact_term_status = JuMP.termination_status(exact_model)

    if exact_term_status != MOI.LOCALLY_SOLVED
        @info "  Warning: node exact model is not solved to local optimality, only to $(exact_term_status). "
    end

    if exact_term_status == MOI.LOCALLY_SOLVED && tree.best_incumbent >= JuMP.objective_value(exact_model)
        @info "  Updating tree best incumbent. Exact model termination status: $(termination_status(exact_model))"
        tree.best_incumbent = JuMP.objective_value(exact_model)
    end
end
=#

function BB.heuristics!(node::BB.JuMPNode)
    ipopt_optimizer = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    root = find_root(node)
    tree = root.auxiliary_data["tree"]
    pm = root.auxiliary_data["PM"]
    exact_model = copy(root.model)
    buses = collect(keys(PM.ref(pm, :bus)))
    JuMP.@variable(exact_model, vr[buses])
    JuMP.@variable(exact_model, vi[buses])

    # get rid of PSDCone constraint
    psdcone = JuMP.all_constraints(exact_model, Array{JuMP.GenericAffExpr{Float64, JuMP.VariableRef}, 1}, MOI.PositiveSemidefiniteConeSquare)[1]
    JuMP.delete(exact_model, psdcone)

    # replace all w variables with voltage product pairs
    wr = PM.var(pm, :wr)
    wr_names = Dict(JuMP.name(var) => i for (i, var) in wr)
    for (i, var) in PM.var(pm, :w)
        wr_names[JuMP.name(var)] = (i, i)
    end
    wi = PM.var(pm, :wi)
    wi_names = Dict(JuMP.name(var) => i for (i, var) in wi)
    # bus_ids = ids(pm, :bus)
    # lookup_w_index = Dict((bi,i) for (i,bi) in enumerate(bus_ids))
    for (functype, settype) in [ (JuMP.GenericAffExpr{Float64,JuMP.VariableRef}, MOI.EqualTo{Float64}),
                                 (JuMP.GenericAffExpr{Float64,JuMP.VariableRef}, MOI.GreaterThan{Float64}),
                                 (JuMP.GenericAffExpr{Float64,JuMP.VariableRef}, MOI.LessThan{Float64})]
        # 
        crefs = JuMP.all_constraints(exact_model, functype, settype)
        for cref in crefs
            cobj = JuMP.constraint_object(cref)
            original_lhs = cobj.func
            new_lhs = zero(JuMP.QuadExpr)
            cref_modified = false
            for (var, coeff) in cobj.func.terms
                var_name = JuMP.name(var)
                if var_name in keys(wr_names)
                    i, j = wr_names[var_name]
                    new_lhs += coeff * (vr[i] * vr[j] + vi[i] * vi[j])
                    original_lhs -= coeff * var
                    cref_modified = true
                elseif var_name in keys(wi_names)
                    i, j = wi_names[var_name]
                    new_lhs += coeff * (vi[i] * vr[j] - vr[i] * vi[j])
                    original_lhs -= coeff * var
                    cref_modified = true
                end
            end
            if cref_modified
                JuMP.drop_zeros!(original_lhs)
                JuMP.@constraint(exact_model, original_lhs + new_lhs in cobj.set)
                JuMP.delete(exact_model, cref)
            end
        end
    end
    for (functype, settype) in [ (JuMP.VariableRef, MOI.GreaterThan{Float64}),
                                 (JuMP.VariableRef, MOI.LessThan{Float64})]
        crefs = JuMP.all_constraints(exact_model, functype, settype)
        for cref in crefs
            cobj = JuMP.constraint_object(cref)
            new_lhs = zero(JuMP.QuadExpr)
            cref_modified = false
            var = cobj.func
            var_name = JuMP.name(var)
            if var_name in PM.keys(wr_names)
                i, j = wr_names[var_name]
                new_lhs = (vr[i] * vr[j] + vi[i] * vi[j])
                cref_modified = true
            elseif var_name in PM.keys(wi_names)
                i, j = wi_names[var_name]
                new_lhs = (vi[i] * vr[j] - vr[i] * vi[j])
                cref_modified = true
            end
            if cref_modified
                JuMP.@constraint(exact_model, new_lhs in cobj.set)
                JuMP.delete(exact_model, cref)
            end    
        end
    end

    # delete W variables
    # for var in all_variables(exact_model)
    #     if occursin("WR", JuMP.name(var)) || occursin("WI", JuMP.name(var))
    #         delete(exact_model, var)
    #     end
    # end

    # convert second order cone constraints into quadratic constraints
    soc_crefs = JuMP.all_constraints(exact_model, Array{JuMP.GenericAffExpr{Float64, JuMP.VariableRef}, 1}, MOI.SecondOrderCone)
    for cref in soc_crefs
        cobj = JuMP.constraint_object(cref)
        terms = cobj.func
        JuMP.@constraint(exact_model, sum(terms[i]^2 for i in 2:length(terms)) <= terms[1]^2)
        JuMP.delete(exact_model, cref)
    end

    # convert rotated second order cone constraints into quadratic constraints (this is needed if objective is quadratic)
    rsoc_crefs = JuMP.all_constraints(exact_model, Array{JuMP.GenericAffExpr{Float64, JuMP.VariableRef}, 1}, MOI.RotatedSecondOrderCone)
    for cref in rsoc_crefs
        cobj = JuMP.constraint_object(cref)
        terms = cobj.func
        JuMP.@constraint(exact_model, sum(terms[i]^2 for i in 3:length(terms)) <= 2 * terms[1].constant * terms[2]) # hacky way to avoid higher order (nonlinear) terms
        JuMP.delete(exact_model, cref)
    end

    # initialize with sdp solution
    # sdp_sol = node.auxiliary_data["node solution"]
    # for (name, val) in sdp_sol
    #     var = JuMP.variable_by_name(exact_model, name)
    #     if !isnothing(var)
    #         JuMP.set_start_value(var, val)
    #     end
    # end

    # initialize with PowerModels default for ACRPowerModel
    JuMP.set_start_value.(vr, 1.0)

    JuMP.set_optimizer(exact_model, ipopt_optimizer)
    JuMP.optimize!(exact_model)
    @info " Heuristics result: $(JuMP.objective_value(exact_model))"
    exact_term_status = JuMP.termination_status(exact_model)

    if exact_term_status != MOI.LOCALLY_SOLVED
        @info "  Warning: node exact model is not solved to local optimality, only to $(exact_term_status). "
    end

    if exact_term_status == MOI.LOCALLY_SOLVED && tree.best_incumbent >= JuMP.objective_value(exact_model)
        @info "  Updating tree best incumbent. Exact model termination status: $(JuMP.termination_status(exact_model))"
        tree.best_incumbent = JuMP.objective_value(exact_model)
    end
end