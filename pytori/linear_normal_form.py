
import numpy as np
import pytori as pt

# Default math library, numpy
_mp = pt.algebra.MathlibNumpy

# UTILITIES
#=========================================================================================
def _to_list(obj):
    """Return obj as a Python list. Works for 1D arrays, SymPy Matrix, lists."""
    return [obj[i] for i in range(len(obj))]

def _abs_float(x, mp=_mp):
    if mp.name == 'sympy':
        import sympy as sp
        return float(sp.Abs(x).evalf())   # force numeric
    return float(mp.abs(x))

def _hstack_columns(cols, mp=_mp):
    """
    Stack a list of column vectors into a matrix:
    - SymPy backend -> return sp.Matrix.hstack(...)
    - otherwise -> return list-of-lists (rows)
    """
    if mp.name == 'sympy':  # columns are SymPy Matrix (n×1)
        import sympy as sp
        return sp.Matrix.hstack(*cols)
    else:
        # plain Python list-of-lists (rows = len(col), cols = len(cols))
        nrows = len(cols[0]) if cols else 0
        ncols = len(cols)
        return np.asarray([[cols[j][i] for j in range(ncols)] for i in range(nrows)])


def _build_M_stacked(U, V,mp=_mp):
    """M = [[U, V], [V*, U*]] in the stacked ordering [Ψ; Ψ*]."""
    
    
    if mp.name == 'sympy':
        import sympy as sp
        n = U.shape[0]
        M = sp.Matrix.zeros(2*n, 2*n)
    else:
        U = np.asarray(U); V = np.asarray(V)
        n = U.shape[0]
        M = np.zeros((2*n, 2*n), dtype=complex)

    M[:n, :n] = U
    M[:n, n:] = V
    M[n:, :n] = mp.conjugate(V)
    M[n:, n:] = mp.conjugate(U)
    return M

def _eigpairs(M, mp=_mp):
    """
    Return (evals, evecs) with evecs as a list of column vectors (len = 2n),
    consistent across SymPy and NumPy backends.
    """
    if mp.name == 'sympy':
        import sympy as sp
        ev = M.eigenvects()
        evals = []
        evecs = []
        for lam, _mult, vecs in ev:
            for v in vecs:
                vv = sp.Matrix(v)
                if vv.shape[1] != 1:  # ensure column
                    vv = vv.T
                evals.append(sp.simplify(lam))
                evecs.append(vv)      # append column vector
        if not evecs:
            raise ValueError("No eigenvectors found (SymPy).")
        return evals, evecs
    else:
        import numpy as np
        vals, vecs = np.linalg.eig(np.asarray(M))
        # vecs has eigenvectors as COLUMNS; convert to list of 1D columns
        evecs = [vecs[:, k] for k in range(vecs.shape[1])]
        return list(vals), evecs


def _angle(z,mp=_mp):
    """Extract angle: """
    zn      = z/mp.abs(z)
    theta   = mp.arctan2(mp.imag(zn), mp.real(zn))  # θ ∈ (-π, π]
    q       = theta / (2.0 * mp.pi)                 # q ∈ (-0.5, 0.5]
    
    if mp.name == 'sympy':
        q = q.evalf()
    return q
#=========================================================================================


# LINEAR NORMAL FORM
#=========================================================================================
def _normalize_eigenvecs(vecs, mp=_mp):

    dim = len(vecs[0])//2

    TN_vals = []
    norms   = []
    for vj in vecs:
        v = _to_list(vj)

        # Splitting into basis components
        a = v[:dim]
        b = [mp.conjugate(_v) for _v in v[dim:]]

        # Σ–norm c = a†a − b†b  (must be > 0 for stable modes)
        a2 = sum(mp.conjugate(ai)*ai for ai in a)
        b2 = sum(mp.conjugate(bi)*bi for bi in b)
        c  = a2 - b2

        # Original norm
        cR = mp.real(c)
        norms.append(cR)

        aN  = [ai/mp.sqrt(mp.abs(cR)) for ai in a]
        bN  = [bi/mp.sqrt(mp.abs(cR)) for bi in b]  

        # reassemble in the same  as input
        if mp.name == 'sympy':
            import sympy as sp
            TN = sp.Matrix(aN + [mp.conjugate(bi) for bi in bN]).evalf()
        else:
            TN = np.asarray(aN + [mp.conjugate(bi) for bi in bN])

        # put back in the same type as input
        TN_vals.append(TN)
    return TN_vals, [int(n/mp.abs(n)) for n in norms]


def _sort_normal_modes(vals, vecs, norms, mp=_mp):
    dim = len(vecs[0]) // 2

    # keep only positive Σ–norm modes
    modes = [(i, vals[i], vecs[i]) for i, sgn in enumerate(norms) if sgn > 0]

    # for each mode, find (anchor_plane, anchor_amp)
    recs = []
    for i, val, vj in modes:
        v = _to_list(vj)
        a, bs = v[:dim], v[dim:]                  # bs already stores b*
        # send to 
        amps = [_abs_float(a[j] + bs[j], mp) for j in range(dim)]

        j_anchor = max(range(dim), key=lambda j: amps[j])
        recs.append((i, val, vj, j_anchor, amps[j_anchor]))

    # greedily pick one mode per plane by descending anchor amplitude
    recs.sort(key=lambda r: r[4], reverse=True)
    picked_idx = [None] * dim
    used_modes = set()
    for i, val, vj, j_anchor, amp in recs:
        if picked_idx[j_anchor] is None:
            picked_idx[j_anchor] = (i, val, vj)
            used_modes.add(i)
        if all(p is not None for p in picked_idx):
            break

    # backfill any missing planes with remaining modes maximizing |s_j|
    remaining = [(i, val, vj) for i, val, vj, _, _ in recs if i not in used_modes]
    for j in range(dim):
        if picked_idx[j] is None:
            best = max(remaining, key=lambda r:  _abs_float(
                _to_list(r[2])[j] + _to_list(r[2])[j+dim]
            ))
            picked_idx[j] = best
            remaining.remove(best)

    vals_sorted = [val for (i, val, vj) in picked_idx]
    vecs_sorted = [vj  for (i, val, vj)  in picked_idx]
    return vals_sorted, vecs_sorted


def _regauge_eigenvecs(vecs, mp=_mp):
    """
    Enforce CS gauge per mode by making s_j = a_j + (b_j)* real-positive
    at the chosen anchor index for that mode.
    anchors: list of indices j_anchor (one per mode), e.g. [0,1] for x,y.
    """
    dim = len(vecs[0])//2

    corrected   = []
    for anchor,vj in enumerate(vecs):
        v = _to_list(vj)
        # Splitting into basis components
        a   = v[:dim]
        bs  = v[dim:] # b*

        # Finding phase reference
        s_anchor = a[anchor] + bs[anchor]    # s_j = a_j + (b_j)*

        # phase to rotate s_anchor → real-positive
        theta = mp.arctan2(mp.imag(s_anchor), mp.real(s_anchor))
        phase = mp.exp(-mp.I*theta)        # multiply entire stacked vector by this
        aG  = [phase*ai     for ai  in a]
        bsG = [phase*bsi    for bsi in bs]    # still b* after gauge

        # reassemble in the same  as input
        if mp.name == 'sympy':
            import sympy as sp
            corrected.append(sp.Matrix(aG + bsG).evalf())
        else:
            corrected.append(np.asarray(aG + bsG))

    return corrected


def M_to_QT(M,mp=_mp):
    vals,vecs   = _eigpairs(M,mp)
    vecs,norms  = _normalize_eigenvecs(vecs,mp)
    vals,vecs   = _sort_normal_modes(vals,vecs,norms,mp)
    vecs        = _regauge_eigenvecs(vecs,mp=mp)


    Qvec   = [_angle(va,mp=mp) for va  in vals]
    Tvec   = [vec              for vec in vecs]

    return Qvec, Tvec


def UV_to_QT(U,V,mp=_mp):
    M = _build_M_stacked(U,V,mp)
    return M_to_QT(M,mp)


def T_to_lambda(Tvec, mp=_mp):

    dim = len(Tvec[0])//2
    
    Lp_cols, Lm_cols = [], []
    for Tj in Tvec:
        T = _to_list(Tj)

        # Splitting into basis components
        a = T[:dim]
        b = [mp.conjugate(_T) for _T in T[dim:]]

        # Build Λ+(:,k) and Λ−(:,k) 
        if mp.name == 'sympy':
            import sympy as sp
            Lp_cols.append(sp.Matrix(a).evalf())
            Lm_cols.append(sp.Matrix(b).evalf()) 
        else:
            Lp_cols.append(a) 
            Lm_cols.append(b)


    Lp = _hstack_columns(Lp_cols, mp)
    Lm = _hstack_columns(Lm_cols, mp)
    return Lp, Lm

def T_to_optics(Tvec,mp=_mp):
    
    dim = len(Tvec[0])//2

    # Ininitialize beta_jk matrices
    alpha= mp.zeros((dim,dim))
    beta = mp.zeros((dim,dim))
    block= mp.zeros((dim,dim))
    
    
    for j in range(dim):
        for k in range(dim):

            s_jk = Tvec[k][j] + Tvec[k][j+dim] # λ+_{jk} + (λ-_{jk})*
            t_jk = Tvec[k][j] - Tvec[k][j+dim] # λ+_{jk} - (λ-_{jk})*

            alpha[j,k] = mp.imag(mp.conjugate(s_jk)*t_jk)
            beta[j,k]  = mp.abs(s_jk)**2
            block[j,k] = mp.real(mp.conjugate(s_jk)*t_jk)
            
    return alpha,beta,block


def T_to_W(Tvec, mp=_mp):
    Lp, Lm = T_to_lambda(Tvec, mp)
    return pt.transforms.lambda_to_W(lambda_plus=Lp, lambda_minus=Lm)


def construct_UV(Qvec=None, lambda_plus=None, lambda_minus=None, W_matrix=None,
                 Lp_list=None,Lm_list=None,W_list=None, mp=_mp):
    """
    Build U and V matrices from Λ± at two locations and phase advances Qvec.
      General (two-location) form:
        U =  Λ2,+ E Λ1,+^†  -  Λ2,- E^* Λ1,-^†
        V = -Λ2,+ E Λ1,-^T  +  Λ2,- E^* Λ1,+^T
      Periodic/same-optics case reduces to:
        U =  Λ+ E Λ+^†  -  Λ- E^* Λ-^†
        V = -Λ+ E Λ-^T  +  Λ- E^* Λ+^T
      with E = diag(exp(i*2π Q_j)).
    """

    
    # We need Qvec !
    assert Qvec is not None, "Qvec (phase advances) must be provided "
    dim = len(Qvec)

    if any(x is not None for x in (lambda_plus, lambda_minus, W_matrix)):
        assert (Lp_list is None) and (Lm_list is None) and (W_list is None), \
            "Provide either (lambda_plus, lambda_minus, W_matrix) or (Lp_list, Lm_list, W_list), not both."
    
        # Resolve λ± (either provided or from W)
        if W_matrix is not None:
            assert lambda_plus is None and lambda_minus is None, \
                "Provide either W_matrix or (lambda_plus, lambda_minus), not both."
            lambda_plus, lambda_minus = pt.transforms.W_to_lambda(W_matrix)

        assert (lambda_plus is not None) and (lambda_minus is not None), \
            "lambda_plus and lambda_minus must be provided (or pass W_matrix)."

        Lp1, Lm1 = lambda_plus, lambda_minus
        Lp2, Lm2 = lambda_plus, lambda_minus

    if any(x is not None for x in (Lp_list, Lm_list, W_list)):
        assert (lambda_plus is None) and (lambda_minus is None) and (W_matrix is None), \
            "Provide either (lambda_plus, lambda_minus, W_matrix) or (Lp_list, Lm_list, W_list), not both."

        if W_list is not None:
            assert len(W_list) == 2, "W_list must be [W1, W2]."
            assert (Lp_list is None) and (Lm_list is None), \
                "Provide either W_list or (Lp_list, Lm_list), not both."
            
            Lp_list = []
            Lm_list = []
            for W in W_list:
                Lp, Lm = pt.transforms.W_to_lambda(W)
                Lp_list.append(Lp)
                Lm_list.append(Lm)

        assert (Lp_list is not None) and (Lm_list is not None), \
            "lambda_plus and lambda_minus must be provided (or pass W_matrix)."
        assert len(Lp_list) == 2 and len(Lm_list) == 2, \
            "Lp_list and Lm_list must each be length 2: [Λ1, Λ2]."

        Lp1, Lm1 = Lp_list[0], Lm_list[0]
        Lp2, Lm2 = Lp_list[1], Lm_list[1]

    assert Lp1.shape == (dim, dim) and Lm1.shape == (dim, dim), \
        f"dimension mismatch, Qvec has length {dim}, but lambda_plus has shape {Lp1.shape}."
    assert Lp2.shape == (dim, dim) and Lm2.shape == (dim, dim), \
        f"dimension mismatch, Qvec has length {dim}, but lambda_plus has shape {Lp2.shape}."

    # Computing rotation matrix E
    E = [mp.exp(2 * mp.I * mp.pi * Qvec[j]) for j in range(dim)]

    # Build U and V explicitly
    U = [[0 for _ in range(dim)] for _ in range(dim)]
    V = [[0 for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for k in range(dim):
            sU = 0
            sV = 0
            for j in range(dim):
                # U = Lp E Lp^† - Lm E^* Lm^†
                sU += Lp2[i, j] * E[j]                  * mp.conjugate(Lp1[k, j])
                sU -= Lm2[i, j] * mp.conjugate(E[j])    * mp.conjugate(Lm1[k, j])
                # V = - Lp E Lm^T + Lm E^* Lp^T)
                sV -= Lp2[i, j] * E[j]                  * Lm1[k, j]
                sV += Lm2[i, j] * mp.conjugate(E[j])    * Lp1[k, j]
            U[i][k] = sU
            V[i][k] = sV
    return U,V