"""High-precision polishing when double-precision NET bound signs are unresolved."""
import time
import numpy as np
import mpmath as mp


def polish_network_face(problem, values, scenarios, probabilities, ridge):
    """Check a fixed face against the original strongly convex objective.

    The Hessian has independent route blocks. Their elimination keeps the
    dense high-precision solve confined to the free reservation variables.
    This is a numerical check, not an interval-arithmetic certificate.
    """
    started=time.perf_counter()
    if ridge<=0 or not np.all(scenarios>0) or not np.all(probabilities>0):
        raise RuntimeError('NET high-precision face check requires positive ridge, demands and weights')
    d,m=problem.decision_dim,problem.scenario_dim;n=len(scenarios)
    upper=np.r_[np.full(d,problem.reservation_upper_bound),np.ones(n*m)]
    low=values<=1e-8;high=upper-values<=1e-8;fixed=low|high;free=~fixed
    iz=np.flatnonzero(free[:d]).tolist();ir=[np.flatnonzero(free[d+k*m:d+(k+1)*m]).tolist() for k in range(n)]
    with mp.workdps(80):
        f=lambda x:mp.mpf(float(x))
        s=[[f(x) for x in row] for row in scenarios]
        weights=[f(x) for x in probabilities];total=mp.fsum(weights);weights=[x/total for x in weights]
        B=[[f(x) for x in row] for row in problem.second_path_incidence]
        D=[[f(x) for x in row] for row in problem.routing_difference]
        unit_incidence=np.all(np.isin(problem.routing_difference,(-1.,0.,1.)))
        integer_D=problem.routing_difference.astype(np.int64) if unit_incidence else None
        D_rows=[[(j,D[l][j]) for j in range(m) if D[l][j]] for l in range(d)]
        B_rows=[[(j,B[l][j]) for j in range(m) if B[l][j]] for l in range(d)]
        D_columns=[[(l,D[l][j]) for l in range(d) if D[l][j]] for j in range(m)]
        column_norms=[mp.fsum(value*value for _,value in entries) for entries in D_columns]
        lam,gamma,eta=f(problem.congestion_penalty),f(problem.routing_penalty),f(ridge)
        bound=[f(x) for x in upper];v=[f(x) for x in values]
        for i in np.flatnonzero(fixed):v[i]=mp.mpf(0) if low[i] else bound[i]
        def system():
            g=[mp.mpf(0) for _ in v]
            for j in range(d):g[j]=f(problem.reservation_costs[j])+eta*(v[j]-f(.1))
            Hzz=mp.eye(len(iz))*eta;blocks=[];hinges=[]
            for k in range(n):
                route=[s[k][j]*v[d+k*m+j] for j in range(m)]
                h=[mp.fsum(value*s[k][j] for j,value in B_rows[l])+mp.fsum(value*route[j] for j,value in D_rows[l])-v[l] for l in range(d)]
                excess=[max(x,mp.mpf(0)) for x in h];active=[x>0 for x in h];w=weights[k]
                for l in range(d):g[l]-=2*w*lam*excess[l]
                for j in range(m):g[d+k*m+j]=2*w*s[k][j]*(lam*mp.fsum(value*excess[l] for l,value in D_columns[j])+gamma*(route[j]-s[k][j]/2))
                for ii,l in enumerate(iz):Hzz[ii,ii]+=2*w*lam*int(active[l])
                js=ir[k]
                gram=integer_D.T@(np.asarray(active,dtype=np.int64)[:,None]*integer_D) if unit_incidence else None
                cross=mp.matrix(len(iz),len(js));Hr=mp.matrix(len(js),len(js))
                for ii,l in enumerate(iz):
                    for jj,j in enumerate(js):cross[ii,jj]=-2*w*lam*int(active[l])*D[l][j]*s[k][j]
                for ii,i in enumerate(js):
                    for jj,j in enumerate(js):Hr[ii,jj]=2*w*s[k][i]*s[k][j]*(lam*(mp.mpf(int(gram[i,j])) if unit_incidence else mp.fsum(D[l][i]*D[l][j] for l in range(d) if active[l]))+gamma*int(i==j))
                blocks.append((js,cross,Hr))
                for l in range(d):
                    lipschitz=int(l in iz)+mp.fsum(abs(D[l][j]*s[k][j]) for j in js)
                    if lipschitz:hinges.append((abs(h[l]),lipschitz))
            return g,Hzz,blocks,hinges
        for _ in range(3):
            g,reduced,blocks,_=system();rhs=mp.matrix(len(iz),1);eliminated=[]
            for ii,j in enumerate(iz):rhs[ii]=-g[j]
            for k,(js,cross,Hr) in enumerate(blocks):
                if not js:eliminated.append(None);continue
                inverse=Hr**-1;gain=inverse*cross.T;offset=inverse*mp.matrix([g[d+k*m+j] for j in js])
                reduced-=cross*gain;rhs+=cross*offset;eliminated.append((gain,offset))
            dz=mp.lu_solve(reduced,rhs) if iz else mp.matrix(0,1)
            for ii,j in enumerate(iz):v[j]+=dz[ii]
            for k,entry in enumerate(eliminated):
                if entry is None:continue
                gain,offset=entry;dr=-offset-gain*dz
                for ii,j in enumerate(ir[k]):v[d+k*m+j]+=dr[ii]
        g,_,_,hinges=system()
        mu=min(eta,min(2*gamma*weights[k]*s[k][j]**2 for k in range(n) for j in range(m)))
        L=eta+2*lam*mp.fsum(weights[k]*(d+mp.fsum(column_norms[j]*s[k][j]**2 for j in range(m))) for k in range(n))+max(2*gamma*weights[k]*s[k][j]**2 for k in range(n) for j in range(m))
        rounding_guard=mp.eps*(1+max(abs(x) for x in v)+max(abs(x) for x in g)+L)*(len(v)+d*m*n+1)**2*10**6
        residual=mp.sqrt(mp.fsum(g[i]**2 for i in np.flatnonzero(free)))
        radius=(residual+rounding_guard)/mu;gradient_error=L*radius+rounding_guard
        slacks=[min(v[i],bound[i]-v[i]) for i in np.flatnonzero(free)]
        duals=[g[i] if low[i] else -g[i] for i in np.flatnonzero(fixed)]
        if (slacks and min(slacks)<=radius) or (duals and min(duals)<=gradient_error):
            raise RuntimeError('NET face remains weak or inconsistent after high-precision polishing')
        if any(margin<=coefficient*radius+rounding_guard for margin,coefficient in hinges):
            raise RuntimeError('NET hinge region remains unresolved after high-precision polishing')
        result=np.array([float(x) for x in v])
        if not np.isfinite(result).all() or np.max(abs(result[:d]-values[:d]))>1e-6:
            raise RuntimeError('NET high-precision polishing changed the decision beyond tolerance')
        report=dict(decimal_digits=80,free_gradient_norm=float(residual),strong_convexity_lower_bound=float(mu),numerical_state_error_bound=str(radius),rounding_guard=str(rounding_guard),minimum_active_multiplier=float(min(duals)) if duals else None,minimum_free_slack=float(min(slacks)) if slacks else None,seconds=time.perf_counter()-started,interpretation='Numerical high-precision checks; not a verified interval-arithmetic bound.')
        return result,free,report
