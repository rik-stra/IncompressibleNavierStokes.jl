
function ke_convection!(setup)
    ##############
    # x-direction

    # differencing matrix
    diag1 = ones(Npx)
    C1D = spdiagm(Npx, Npx + 1, 0 => -diag1, 1 => diag1)
    Ckx = kron(sparse(I, Npy, Npy), C1D)

    # interpolating u
    I1D = sparse(I, Npx + 1, Npx + 1)

    # boundary conditions
    B1Du, Btempu, ybcl, ybcr =
        BC_general(Npx + 1, Nux_in, Npx + 1 - Nux_in, BC.u.left, BC.u.right, hx[1], hx[end])
    mat_hy = spdiagm(hy)
    uLe_i = interp1(y, uLe, yp)
    uRi_i = interp1(y, uRi, yp)
    ybc = kron(uLe_i, ybcl) + kron(uRi_i, ybcr)
    yIu_kx = kron(mat_hy, I1D * Btempu) * ybc
    Iu_kx = kron(mat_hy, I1D * B1Du)

    # averaging k
    # already constructed in ke_production!
    diag2 = 0.5 * ones(Npx + 1)
    A1D = spdiagm([diag2 diag2], [0 1], Npx + 1, Npx + 2)

    # boundary conditions
    B1D, Btemp, ybcl, ybcr =
        BC_general_stag(Npx + 2, Npx, 2, BC.k.left, BC.k.right, hx[1], hx[end])
    ybc = kron(kLe, ybcl) + kron(kRi, ybcr)
    yAk_kx = kron(speye(Npy), A1D * Btemp) * ybc
    Ak_kx = kron(speye(Npy), A1D * B1D)


    ##############
    # y-direction

    # differencing matrix
    diag1 = ones(Npy)
    C1D = spdiagm(Npy, Npy + 1, 0 => -diag1, 1 => diag1)
    Cky = kron(C1D, sparse(I, Npx, Npx))

    # interpolating v
    I1D = sparse(I, Npy + 1, Npy + 1)

    # boundary conditions
    B1Dv, Btempv, ybcl, ybcu =
        BC_general(Npy + 1, Nvy_in, Npy + 1 - Nvy_in, BC.v.low, BC.v.up, hy[1], hy[end])
    mat_hx = spdiagm(hx, 0, Npx, Npx)
    vLo_i = interp1(x, vLo, xp)
    vUp_i = interp1(x, vUp, xp)
    ybc = kron(ybcl, vLo_i) + kron(ybcu, vUp_i)
    yIv_ky = kron(I1D * Btempv, mat_hx) * ybc
    Iv_ky = kron(I1D * B1Dv, mat_hx)

    # averaging k
    diag2 = 0.5 * ones(Npy + 1)
    A1D = spdiagm(Npy + 1, Npy + 2, 0 => diag2, 1 => diag2)

    # boundary conditions
    B1D, Btemp, ybcl, ybcr =
        BC_general_stag(Npy + 2, Npy, 2, BC.k.low, BC.k.up, hy[1], hy[end])
    ybc = kron(ybcl, kLo) + kron(ybcr, kUp)
    yAk_ky = kron(A1D * Btemp, speye(Npx)) * ybc
    Ak_ky = kron(A1D * B1D, speye(Npx))
end
