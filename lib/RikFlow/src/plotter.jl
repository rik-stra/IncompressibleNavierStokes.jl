using CairoMakie

vortplot(state; setup) = begin
    state isa Observable || (state = Observable(state))
    ω = lift(state) do state
        vx = mean(state.u[1],dims=3)[1:end-1, :]
        vy = mean(state.u[2],dims=3)[:, 1:end-1]
        ω = -diff(vx; dims = 2) + diff(vy; dims = 1)
        Array(ω)
    end
    heatmap(ω; figure = (; size = (900, 350)), axis = (; aspect = DataAspect()))
end