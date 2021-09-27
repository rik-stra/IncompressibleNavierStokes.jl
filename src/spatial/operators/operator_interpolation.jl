"""
Construct averaging operators.
"""
function operator_interpolation!(setup)

    # boundary conditions
    BC = setup.BC

    Nx = setup.grid.Nx
    Ny = setup.grid.Ny

    # number of interior points and boundary points
    @unpack Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t = setup.grid
    @unpack Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t = setup.grid
    @unpack hx, hy, hxi, hyi = setup.grid
    @unpack Buvy, Bvux = setup.grid

    order4 = setup.discretization.order4

    if order4
        beta = setup.discretization.beta
        @unpack hxi3, hyi3, hx3, hy3 = setup.grid
    end

    weight = 1 / 2

    ##

    mat_hx = spdiagm(Nx, Nx, hxi)
    mat_hy = spdiagm(Ny, Ny, hyi)

    # periodic boundary conditions
    if BC.u.left == "per" && BC.u.right == "per"
        mat_hx2 = spdiagm(Nx + 2, Nx + 2, [hx[end]; hx; hx[1]])
    else
        mat_hx2 = spdiagm(Nx + 2, Nx + 2, [hx[1]; hx; hx[end]])
    end

    if BC.v.low == "per" && BC.v.up == "per"
        mat_hy2 = spdiagm(Ny + 2, Ny + 2, [hy[end]; hy; hy[1]])
    else
        mat_hy2 = spdiagm(Ny + 2, Ny + 2, [hy[1]; hy; hy[end]])
    end


    ## Interpolation operators, u-component
    if order4
        mat_hx3 = spdiagm(Nx, Nx, hxi3)
        mat_hy3 = spdiagm(Ny, Ny, hyi3)

        weight1 = 1 / 2 * beta
        weight2 = 1 / 2 * (1 - beta)

        # periodic boundary conditions
        if BC.u.left == "per" && BC.u.right == "per"
            mat_hx2 = spdiagm(Nx + 4, Nx + 4, [hx[end-1]; hx[end]; hx; hx[1]; hx[2]])
            mat_hx4 = spdiagm(Nx + 4, Nx + 4, [hx3[end-1]; hx3[end]; hxi3; hx3[1]; hx3[2]])
        else
            mat_hx2 = spdiagm(Nx + 4, Nx + 4, [hx[2]; hx[1]; hx; hx[end]; hx[end-1]])
            mat_hx4 = spdiagm(
                Nx + 4,
                Nx + 4,
                [
                    hx[1] + hx[2] + hx[3]
                    2 * hx[1] + hx[2]
                    hxi3
                    2 * hx[end] + hx[end-1]
                    hx[end] + hx[end-1] + hx[end-2]
                ],
            )
        end

        if BC.v.low == "per" && BC.v.up == "per"
            mat_hy2 = spdiagm(Ny + 4, Ny + 4, [hy[end-1]; hy[end]; hy; hy[1]; hy[2]])
            mat_hy4 = spdiagm(Ny + 4, Ny + 4, [hy3[end-1]; hy3[end]; hyi3; hy3[1]; hy3[2]])
        else
            mat_hy2 = spdiagm(Ny + 4, Ny + 4, [hy[2]; hy[1]; hy; hy[end]; hy[end-1]])
            mat_hy4 = spdiagm(
                Ny + 4,
                Ny + 4,
                [
                    hy[1] + hy[2] + hy[3]
                    2 * hy[1] + hy[2]
                    hyi3
                    2 * hy[end] + hy[end-1]
                    hy[end] + hy[end-1] + hy[end-2]
                ],
            )
        end

        ## Iu_ux
        diag1 = weight1 * ones(Nux_t + 1)
        diag2 = weight2 * ones(Nux_t + 1)
        I1D = spdiagm(Nux_t - 1, Nux_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # boundary conditions
        Iu_ux_BC = BC_int2(
            Nux_t + 2,
            Nux_in,
            Nux_t + 2 - Nux_in,
            BC.u.left,
            BC.u.right,
            hx[1],
            hx[end],
        )

        # extend to 2D
        Iu_ux = kron(mat_hy, I1D * Iu_ux_BC.B1D)
        Iu_ux_BC.Bbc = kron(mat_hy, I1D * Iu_ux_BC.Btemp)

        ## Iu_ux3
        diag1 = weight1 * ones(Nux_t + 4)
        diag2 = weight2 * ones(Nux_t + 4)
        I1D3 =
            spdiagm(Nux_in + 3, Nux_t + 4, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # boundary conditions
        Iu_ux_BC3 = BC_int3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            BC.u.left,
            BC.u.right,
            hx[1],
            hx[end],
        )
        # extend to 2D
        Iu_ux3 = kron(mat_hy3, I1D3 * Iu_ux_BC3.B1D)
        Iu_ux_BC3.Bbc = kron(mat_hy3, I1D3 * Iu_ux_BC3.Btemp)

        ## Iv_uy
        diag1 = weight1 * ones(Nvx_t)
        diag2 = weight2 * ones(Nvx_t)
        I1D = spdiagm(Nvx_t - 1, Nvx_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # restrict to u-points
        # the restriction is essentially 1D so it can be directly applied to I1D
        I1D = Bvux * I1D * mat_hx2
        I2D = kron(sparse(I, Nuy_t - 1, Nuy_t - 1), I1D)
        # boundary conditions low/up
        Nb = Nuy_in + 1 - Nvy_in
        Iv_uy_BC_lu = BC_general(Nuy_in + 1, Nvy_in, Nb, BC.v.low, BC.v.up, hy[1], hy[end])
        Iv_uy_BC_lu.B2D = kron(Iv_uy_BC_lu.B1D, sparse(I, Nvx_in, Nvx_in))
        Iv_uy_BC_lu.Bbc = kron(Iv_uy_BC_lu.Btemp, sparse(I, Nvx_in, Nvx_in))
        # boundary conditions left/right
        Iv_uy_BC_lr = BC_int_mixed_stag2(
            Nvx_t + 2,
            Nvx_in,
            Nvx_t + 2 - Nvx_in,
            BC.v.left,
            BC.v.right,
            hx[1],
            hx[end],
        )
        # take I2D into left/right operators for convenience
        Iv_uy_BC_lr.B2D = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_BC_lr.B1D)
        Iv_uy_BC_lr.Bbc = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_BC_lr.Btemp)
        # resulting operator:
        Iv_uy = Iv_uy_BC_lr.B2D * Iv_uy_BC_lu.B2D

        ## Iv_uy3
        diag1 = weight1 * ones(Nvx_t)
        diag2 = weight2 * ones(Nvx_t)
        I1D = spdiagm(Nvx_t - 1, Nvx_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # restrict to u-points
        # the restriction is essentially 1D so it can be directly applied to I1D
        I1D = Bvux * I1D * mat_hx4
        I2D = kron(sparse(I, Nuy_t + 1, Nuy_t + 1), I1D)
        # boundary conditions low/up
        Nb = Nuy_in + 3 - Nvy_in
        Iv_uy_BC_lu3 =
            BC_int_mixed2(Nuy_in + 3, Nvy_in, Nb, BC.v.low, BC.v.up, hy[1], hy[end])
        Iv_uy_BC_lu3.B2D = kron(Iv_uy_BC_lu3.B1D, sparse(I, Nvx_in, Nvx_in))
        Iv_uy_BC_lu3.Bbc = kron(Iv_uy_BC_lu3.Btemp, sparse(I, Nvx_in, Nvx_in))
        # boundary conditions left/right
        Iv_uy_BC_lr3 = BC_int_mixed_stag3(
            Nvx_t + 2,
            Nvx_in,
            Nvx_t + 2 - Nvx_in,
            BC.v.left,
            BC.v.right,
            hx[1],
            hx[end],
        )
        # take I2D into left/right operators for convenience
        Iv_uy_BC_lr3.B2D = I2D * kron(sparse(I, Nuy_t + 1, Nuy_t + 1), Iv_uy_BC_lr3.B1D)
        Iv_uy_BC_lr3.Bbc = I2D * kron(sparse(I, Nuy_t + 1, Nuy_t + 1), Iv_uy_BC_lr3.Btemp)
        # resulting operator:
        Iv_uy3 = Iv_uy_BC_lr3.B2D * Iv_uy_BC_lu3.B2D

        ## Iu_vx
        diag1 = weight1 * ones(Nuy_t)
        diag2 = weight2 * ones(Nuy_t)
        I1D = spdiagm(Nuy_t - 1, Nuy_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # restrict to v-points
        I1D = Buvy * I1D * mat_hy2
        I2D = kron(I1D, sparse(I, Nvx_t - 1, Nvx_t - 1))
        # boundary conditions low/up
        Iu_vx_BC_lu = BC_int_mixed_stag2(
            Nuy_t + 2,
            Nuy_in,
            Nuy_t + 2 - Nuy_in,
            BC.u.low,
            BC.u.up,
            hy[1],
            hy[end],
        )
        Iu_vx_BC_lu.B2D = I2D * kron(Iu_vx_BC_lu.B1D, sparse(I, Nvx_t - 1, Nvx_t - 1))
        Iu_vx_BC_lu.Bbc = I2D * kron(Iu_vx_BC_lu.Btemp, sparse(I, Nvx_t - 1, Nvx_t - 1))

        # boundary conditions left/right
        Nb = Nvx_in + 1 - Nux_in
        Iu_vx_BC_lr =
            BC_general(Nvx_in + 1, Nux_in, Nb, BC.u.left, BC.u.right, hx[1], hx[end])

        Iu_vx_BC_lr.B2D = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_BC_lr.B1D)
        Iu_vx_BC_lr.Bbc = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_BC_lr.Btemp)

        # resulting operator:
        Iu_vx = Iu_vx_BC_lu.B2D * Iu_vx_BC_lr.B2D

        ## Iu_vx3
        diag1 = weight1 * ones(Nuy_t)
        diag2 = weight2 * ones(Nuy_t)
        I1D = spdiagm(Nuy_t - 1, Nuy_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # restrict to v-points
        I1D = Buvy * I1D * mat_hy4
        I2D = kron(I1D, sparse(I, Nvx_t + 1, Nvx_t + 1))
        # boundary conditions low/up
        Iu_vx_BC_lu3 = BC_int_mixed_stag3(
            Nuy_t + 2,
            Nuy_in,
            Nuy_t + 2 - Nuy_in,
            BC.u.low,
            BC.u.up,
            hy[1],
            hy[end],
        )
        Iu_vx_BC_lu3.B2D = I2D * kron(Iu_vx_BC_lu3.B1D, sparse(I, Nvx_t + 1, Nvx_t + 1))
        Iu_vx_BC_lu3.Bbc = I2D * kron(Iu_vx_BC_lu3.Btemp, sparse(I, Nvx_t + 1, Nvx_t + 1))

        # boundary conditions left/right
        Nb = Nvx_in + 3 - Nux_in
        Iu_vx_BC_lr3 =
            BC_int_mixed2(Nvx_in + 3, Nux_in, Nb, BC.u.left, BC.u.right, hx[1], hx[end])

        Iu_vx_BC_lr3.B2D = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_BC_lr3.B1D)
        Iu_vx_BC_lr3.Bbc = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_BC_lr3.Btemp)

        # resulting operator:
        Iu_vx3 = Iu_vx_BC_lu3.B2D * Iu_vx_BC_lr3.B2D

        ## Iv_vy
        diag1 = weight1 * ones(Nvy_t + 1, 1)
        diag2 = weight2 * ones(Nvy_t + 1, 1)
        I1D = spdiagm(Nvy_t - 1, Nvy_t + 2, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)

        # boundary conditions
        Iv_vy_BC = BC_int2(
            Nvy_t + 2,
            Nvy_in,
            Nvy_t + 2 - Nvy_in,
            BC.v.low,
            BC.v.up,
            hy[1],
            hy[end],
        )

        # extend to 2D
        Iv_vy = kron(I1D * Iv_vy_BC.B1D, mat_hx)
        Iv_vy_BC.Bbc = kron(I1D * Iv_vy_BC.Btemp, mat_hx)

        ## Iv_vy3
        diag1 = weight1 * ones(Nvy_t + 4)
        diag2 = weight2 * ones(Nvy_t + 4)
        I1D3 =
            spdiagm(Nvy_in + 3, Nvy_t + 4, 0 => diag2, 1 => diag1, 2 => diag1, 3 => diag2)
        # boundary conditions
        Iv_vy_BC3 = BC_int3(
            Nvy_t + 4,
            Nvy_in,
            Nvy_t + 4 - Nvy_in,
            BC.v.low,
            BC.v.up,
            hy[1],
            hy[end],
        )
        # extend to 2D
        Iv_vy3 = kron(I1D3 * Iv_vy_BC3.B1D, mat_hx3)
        Iv_vy_BC3.Bbc = kron(I1D3 * Iv_vy_BC3.Btemp, mat_hx3)
    else
        ## Iu_ux
        diag1 = weight * ones(Nux_t)
        I1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)
        # boundary conditions
        Iu_ux_BC = BC_general(Nux_t, Nux_in, Nux_b, BC.u.left, BC.u.right, hx[1], hx[end])

        # extend to 2D
        Iu_ux = kron(mat_hy, I1D * Iu_ux_BC.B1D)
        Iu_ux_BC.Bbc = kron(mat_hy, I1D * Iu_ux_BC.Btemp)


        ## Iv_uy
        diag1 = weight * ones(Nvx_t)
        I1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)
        # the restriction is essentially 1D so it can be directly applied to I1D
        I1D = Bvux * I1D * mat_hx2
        I2D = kron(sparse(I, Nuy_t - 1, Nuy_t - 1), I1D)


        # boundary conditions low/up
        Nb = Nuy_in + 1 - Nvy_in
        Iv_uy_BC_lu = BC_general(Nuy_in + 1, Nvy_in, Nb, BC.v.low, BC.v.up, hy[1], hy[end])
        Iv_uy_BC_lu.B2D = kron(Iv_uy_BC_lu.B1D, sparse(I, Nvx_in))
        Iv_uy_BC_lu.Bbc = kron(Iv_uy_BC_lu.Btemp, sparse(I, Nvx_in))


        # boundary conditions left/right
        Iv_uy_BC_lr =
            BC_general_stag(Nvx_t, Nvx_in, Nvx_b, BC.v.left, BC.v.right, hx[1], hx[end])
        # take I2D into left/right operators for convenience
        Iv_uy_BC_lr.B2D = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_BC_lr.B1D)
        Iv_uy_BC_lr.Bbc = I2D * kron(sparse(I, Nuy_t - 1, Nuy_t - 1), Iv_uy_BC_lr.Btemp)

        # resulting operator:
        Iv_uy = Iv_uy_BC_lr.B2D * Iv_uy_BC_lu.B2D


        ## Interpolation operators, v-component

        ## Iu_vx
        diag1 = weight * ones(Nuy_t)
        I1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)
        I1D = Buvy * I1D * mat_hy2
        I2D = kron(I1D, sparse(I, Nvx_t - 1, Nvx_t - 1))

        # boundary conditions low/up
        Iu_vx_BC_lu =
            BC_general_stag(Nuy_t, Nuy_in, Nuy_b, BC.u.low, BC.u.up, hy[1], hy[end])
        Iu_vx_BC_lu.B2D = I2D * kron(Iu_vx_BC_lu.B1D, sparse(I, Nvx_t - 1, Nvx_t - 1))
        Iu_vx_BC_lu.Bbc = I2D * kron(Iu_vx_BC_lu.Btemp, sparse(I, Nvx_t - 1, Nvx_t - 1))

        # boundary conditions left/right
        Nb = Nvx_in + 1 - Nux_in
        Iu_vx_BC_lr =
            BC_general(Nvx_in + 1, Nux_in, Nb, BC.u.left, BC.u.right, hx[1], hx[end])

        Iu_vx_BC_lr.B2D = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_BC_lr.B1D)
        Iu_vx_BC_lr.Bbc = kron(sparse(I, Nuy_in, Nuy_in), Iu_vx_BC_lr.Btemp)

        # resulting operator:
        Iu_vx = Iu_vx_BC_lu.B2D * Iu_vx_BC_lr.B2D


        ## Iv_vy
        diag1 = weight * ones(Nvy_t)
        I1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)
        # boundary conditions
        Iv_vy_BC = BC_general(Nvy_t, Nvy_in, Nvy_b, BC.v.low, BC.v.up, hy[1], hy[end])

        # extend to 2D
        Iv_vy = kron(I1D * Iv_vy_BC.B1D, mat_hx)
        Iv_vy_BC.Bbc = kron(I1D * Iv_vy_BC.Btemp, mat_hx)
    end

    ## store in setup structure
    setup.discretization.Iu_ux = Iu_ux
    setup.discretization.Iv_uy = Iv_uy
    setup.discretization.Iu_vx = Iu_vx
    setup.discretization.Iv_vy = Iv_vy

    setup.discretization.Iu_ux_BC = Iu_ux_BC
    setup.discretization.Iv_vy_BC = Iv_vy_BC

    setup.discretization.Iv_uy_BC_lr = Iv_uy_BC_lr
    setup.discretization.Iv_uy_BC_lu = Iv_uy_BC_lu
    setup.discretization.Iu_vx_BC_lr = Iu_vx_BC_lr
    setup.discretization.Iu_vx_BC_lu = Iu_vx_BC_lu

    if order4
        setup.discretization.Iu_ux3 = Iu_ux3
        setup.discretization.Iv_uy3 = Iv_uy3
        setup.discretization.Iu_vx3 = Iu_vx3
        setup.discretization.Iv_vy3 = Iv_vy3
        setup.discretization.Iu_ux_BC3 = Iu_ux_BC3
        setup.discretization.Iv_uy_BC_lr3 = Iv_uy_BC_lr3
        setup.discretization.Iv_uy_BC_lu3 = Iv_uy_BC_lu3
        setup.discretization.Iu_vx_BC_lr3 = Iu_vx_BC_lr3
        setup.discretization.Iu_vx_BC_lu3 = Iu_vx_BC_lu3
        setup.discretization.Iv_vy_BC3 = Iv_vy_BC3
    end

    setup
end
