###########################################################################
#                          PowerModels Extensions                         #
###########################################################################

# get rid of angle difference bounds, which will be included in other parts of code
# the valid inequality cuts for voltage products are still kept
function PM.constraint_voltage_angle_difference(pm::PM.AbstractWModels, n::Int, f_idx, angmin, angmax)
    i, f_bus, t_bus = f_idx

    w_fr = PM.var(pm, n, :w, f_bus)
    w_to = PM.var(pm, n, :w, t_bus)
    wr   = PM.var(pm, n, :wr, (f_bus, t_bus))
    wi   = PM.var(pm, n, :wi, (f_bus, t_bus))

    PM.cut_complex_product_and_angle_difference(pm.model, w_fr, w_to, wr, wi, angmin, angmax)
end

# modified from original objective_min_fuel_and_flow_cost to allow for
# the case with no generators
function objective_min_fuel_and_flow_cost_mod(pm::PM.AbstractPowerModel; kwargs...)
    model = PM.check_cost_models(pm)
    if model == 1
        return PM.objective_min_fuel_and_flow_cost_pwl(pm; kwargs...)
    elseif model == 2
        return PM.objective_min_fuel_and_flow_cost_polynomial(pm; kwargs...)
    elseif model === nothing
        return JuMP.@objective(pm.model, Min, 0)
    else
        Memento.error(_LOGGER, "Only cost models of types 1 and 2 are supported at this time, given cost model type of $(model)")
    end
end

# This is essentially copied from original PM.build_opf 
function PM.build_opf(pm::NodeWRMPowerModel)
    PM.variable_bus_voltage(pm)
    PM.variable_gen_power(pm)
    PM.variable_branch_power(pm)
    PM.variable_dcline_power(pm)

    objective_min_fuel_and_flow_cost_mod(pm)

    PM.constraint_model_voltage(pm)

    :cut_bus in keys(PM.ref(pm)) ? cut_bus = PM.ids(pm, :cut_bus) : cut_bus = [] # This is needed in order to be compatible with network decomposition later
    for i in setdiff(PM.ids(pm, :bus), cut_bus)
        PM.constraint_power_balance(pm, i)
    end

    for i in PM.ids(pm, :branch)
        PM.constraint_ohms_yt_from(pm, i)
        PM.constraint_ohms_yt_to(pm, i)
        PM.constraint_voltage_angle_difference(pm, i)
        PM.constraint_thermal_limit_from(pm, i)
        PM.constraint_thermal_limit_to(pm, i)
    end

    for i in PM.ids(pm, :dcline)
        PM.constraint_dcline_power_losses(pm, i)
    end
end
