###########################################################################
#                          Initialization Method                          #
###########################################################################

function initialize(pm::NodeWRMPowerModel, file::String)::Tuple{BB.AbstractTree, BB.AbstractNode}
    # collect data
    Lii = Dict(i => bus["vmin"]^2 for (i, bus) in PM.ref(pm, :bus))
    Uii = Dict(i => bus["vmax"]^2 for (i, bus) in PM.ref(pm, :bus))
    Lij = Dict((i,j) => tan(branch["angmin"]) for ((i,j), branch) in PM.ref(pm, :buspairs))
    Uij = Dict((i,j) => tan(branch["angmax"]) for ((i,j), branch) in PM.ref(pm, :buspairs))

    # initialize branch-and-cut tree
    node = BB.JuMPNode{SpatialBCBranch}(pm.model)
    node.auxiliary_data["best_id"] = 0
    node.auxiliary_data["best_bound"] = -Inf
    node.auxiliary_data["PM"] = pm
    node.auxiliary_data["Lii"] = Lii
    node.auxiliary_data["Uii"] = Uii
    node.auxiliary_data["Lij"] = Lij
    node.auxiliary_data["Uij"] = Uij
    node.auxiliary_data["Cuts"] = Dict( "angle_lb" => Dict() , "angle_ub" => Dict())
    wr = PM.var(pm, :wr)
    wi = PM.var(pm, :wi)
    for (i, branch) in PM.ref(pm, :branch)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        node.auxiliary_data["Cuts"]["angle_lb"][(f_bus, t_bus)] = JuMP.@constraint(pm.model, Lij[(f_bus, t_bus)] * wr[(f_bus, t_bus)] <= wi[(f_bus, t_bus)])
        node.auxiliary_data["Cuts"]["angle_ub"][(f_bus, t_bus)] = JuMP.@constraint(pm.model, Uij[(f_bus, t_bus)] * wr[(f_bus, t_bus)] >= wi[(f_bus, t_bus)])
    end

    node.auxiliary_data["best_bounds"] = []
    node.auxiliary_data["rank1"] = Dict() # record all rank-1 solutions encountered

    tree = BB.initialize_tree(node)
    node.auxiliary_data["tree"] = tree

    # set incumbent
    ipopt_optimizer = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    ipopt_solution = PM.run_opf(file, PM.ACRPowerModel, ipopt_optimizer)
    tree.best_incumbent = ipopt_solution["objective"]

    return tree, node
end
