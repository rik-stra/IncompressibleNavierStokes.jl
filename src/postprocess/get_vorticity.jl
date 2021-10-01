function get_vorticity(V, t, setup)
    # Vorticity values at pressure midpoints
    # This should be consistent with operator_postprocessing
    @unpack Nu, Nv, Nux_in, Nvy_in, Nx, Ny = setup.grid
    @unpack Wv_vx, Wu_uy = setup.discretization

    uₕ = @view V[1:Nu]
    vₕ = @view V[Nu+1:Nu+Nv]

    if setup.bc.u.left == "per" && setup.bc.v.low == "per"
        uₕ_in = uₕ
        vₕ_in = vₕ
    else
        # Velocity at inner points
        diagpos = 0
        if setup.bc.u.left == "pres"
            diagpos = 1
        end
        if setup.bc.u.right == "per" && setup.bc.u.left == "per"
            # Like pressure left
            diagpos = 1
        end

        B1D = spdiagm(Nx - 1, Nux_in, diagpos => ones(Nx - 1))
        B2D = kron(sparse(I, Ny, Ny), B1D)

        uₕ_in = B2D * uₕ

        diagpos = 0
        if setup.bc.v.low == "pres"
            diagpos = 1
        end
        if setup.bc.v.low == "per" && setup.bc.v.up == "per"
            # Like pressure low
            diagpos = 1
        end

        B1D = spdiagm(Ny - 1, Nvy_in, diagpos => ones(Ny - 1))
        B2D = kron(B1D, sparse(I, Nx, Nx))

        vₕ_in = B2D * vₕ
    end

    Wv_vx * vₕ_in - Wu_uy * uₕ_in
end
