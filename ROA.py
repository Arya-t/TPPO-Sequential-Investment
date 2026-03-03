from Region_Generator import *
from sklearn.linear_model import LinearRegression
from numpy.polynomial.hermite import hermval
from itertools import combinations
import os

DEFAULT_SEED = 20240908
        
def sequence_generation(H, k, T):
    all_options = set(range(1, H + 1))

    portfolios = []
    for i in range(1, k + 1):
        portfolios.extend(combinations(all_options, i))
    portfolios = [frozenset(p) for p in portfolios if len(p) <= k]
    portfolios = list(set(portfolios))

    def generate_sequences(remaining_options, current_sequence, depth=0):
    
        if depth == T:
            if not remaining_options:
                return [[list(p) for p in current_sequence]]
            return []

        if not remaining_options:
            return [[list(p) for p in current_sequence]]

        sequences = []
        for portfolio in portfolios:
            if len(portfolio) > k:
                continue

            if portfolio.issubset(remaining_options):
                sequences.extend(
                    generate_sequences(
                        remaining_options - portfolio,
                        current_sequence + [portfolio],
                        depth + 1
                    )
                )
        return sequences

    valid_sequences = generate_sequences(all_options, [], 0)

    final = []
    for seq in valid_sequences:
        included = set()
        for portfolio in seq:
            included.update(portfolio)
        if included == all_options:
            final.append(seq)

    return final

def random_sequence_generation(H, k, m, T):
    all_options = set(range(1, H + 1))
    portfolios = []
    for i in range(1, k + 1):
        portfolios.extend(combinations(all_options, i))
    
    portfolios = [frozenset(p) for p in portfolios]
    portfolios = list(set(portfolios))
    
    sequences = []
    attempts = 0
    max_attempts = m * 10  
    
    while len(sequences) < m and attempts < max_attempts:
        attempts += 1
        remaining_options = all_options.copy()
        current_sequence = []
        
        for _ in range(T+1):
            if not remaining_options:
                break
                
            valid_portfolios = [p for p in portfolios if p.issubset(remaining_options)]
            if not valid_portfolios:
                break
                
            selected_portfolio = random.choice(valid_portfolios)
            current_sequence.append(selected_portfolio)
            remaining_options -= selected_portfolio
        
        included_options = set()
        for portfolio in current_sequence:
            included_options.update(portfolio)
            
        if included_options == all_options:
            converted_sequence = [list(combo) for combo in current_sequence]
            sequences.append(converted_sequence)
    
    if len(sequences) < m:
        print(f"Warning: Ony {len(sequences)} sequences generated，less than {m}")
    
    return sequences
        

def _extract_base_seed(rng):
    try:
        st = rng.bit_generator.state
        if isinstance(st, dict):
            if "state" in st and isinstance(st["state"], dict) and "state" in st["state"]:
                return int(st["state"]["state"]) & 0xFFFFFFFF
            if "state" in st and isinstance(st["state"], (int, np.integer)):
                return int(st["state"]) & 0xFFFFFFFF
        return 20240908
    except Exception:
        return 20240908

def _rng_from_key(base_seed, path_id, origin_id, dest_id, t, stream=0):
    ss = np.random.SeedSequence([int(base_seed), int(path_id), int(origin_id), int(dest_id), int(t), int(stream)])
    return np.random.default_rng(ss)
    
def calculate_od_demand(
    t,
    L,
    region,
    dest_id,
    distribution='gamma',
    rng=None,
    path_id=0
):

    if rng is None:
        rng = np.random.default_rng(20240908)

    base_seed = _extract_base_seed(rng)
    j_idx = dest_id - 1  # dest_id: 1..N -> 0..N-1

    if not hasattr(region, "od_demand_paths"):
        raise AttributeError("region must have attribute od_demand_paths (list of 2D arrays).")
    if path_id < 0 or path_id >= len(region.od_demand_paths):
        raise IndexError(f"path_id out of range: {path_id}")

    od_mat = region.od_demand_paths[path_id]

    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    if j_idx < 0 or j_idx >= od_mat.shape[1]:
        raise IndexError(f"dest_id out of range: dest_id={dest_id}, j_idx={j_idx}")

    if t == 0:
        q0 = od_mat[0, j_idx]
        if np.isnan(q0):
            raise ValueError(
                f"Q_ij,0 is NaN (not initialized). region.id={region.id}, dest_id={dest_id}, path_id={path_id}"
            )
        return float(q0)

    qt = od_mat[t, j_idx]
    if not np.isnan(qt):
        return float(qt)

    prev_q = od_mat[t - 1, j_idx]
    if np.isnan(prev_q):
        prev_q = calculate_od_demand(
            t - 1, L, region, dest_id,
            distribution=distribution, rng=rng, path_id=path_id
        )

    mu     = float(getattr(region, 'mu', 0.05))
    sigma  = float(getattr(region, 'sigma', 0.2))
    alpha  = float(getattr(region, 'alpha', 2.0))
    beta   = float(getattr(region, 'beta', 1.0))
    lam    = float(getattr(region, 'lambda_', 0.1))
    dt = 1.0

    L_eff = max(1, int(L))

    kappa = float(getattr(region, "kappa", 1.0))
    scale_mode = getattr(region, "scale_mode", "nonstationary")
    scale_a = float(getattr(region, "scale_a", 0.0))

    if scale_mode == "stationary":
        f_I = 1.0
    else:
        # nonstationary: 1 + a log(1+I)
        f_I = 1.0 + scale_a * np.log(1.0 + L_eff)

    i_id = int(region.id)

    rng_spill = _rng_from_key(base_seed, path_id, i_id, dest_id, t, stream=1)
    rng_jump  = _rng_from_key(base_seed, path_id, i_id, dest_id, t, stream=2)
    rng_gbm   = _rng_from_key(base_seed, path_id, i_id, dest_id, t, stream=3)

    m = alpha * beta
    v = alpha * (beta ** 2)
    v = max(v, 1e-12)

    if distribution == 'gamma':
        spill = rng_spill.gamma(shape=alpha, scale=beta)

    elif distribution == 'normal':
        spill = rng_spill.normal(loc=m, scale=np.sqrt(v))

    elif distribution == 'laplace':
        b = np.sqrt(v / 2.0)
        spill = rng_spill.laplace(loc=m, scale=b)

    elif distribution == 'lognormal':
        m_pos = max(m, 1e-8)
        sigma2_ln = np.log(1.0 + v / (m_pos ** 2))
        sigma_ln = np.sqrt(max(sigma2_ln, 1e-12))
        mu_ln = np.log(m_pos) - 0.5 * sigma2_ln
        spill = rng_spill.lognormal(mean=mu_ln, sigma=sigma_ln)

    else:
        spill = 0.0

    jump_happens = (rng_jump.poisson(lam * dt) > 0)
    J = (kappa * spill * f_I) if jump_happens else 0.0

    Z = rng_gbm.normal(0.0, 1.0)
    growth = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J)
    new_q = float(prev_q) * float(growth)

    od_mat[t, j_idx] = new_q
    return new_q

def strike_price(region_dict, p1=1.0, p2=1.0, rho=None):
    demand_ii = []
    demand_ij = []

    for region_id, region in region_dict.items():
        i_idx = region_id - 1  # region_id: 1..N

        if not hasattr(region, "od_demand"):
            raise AttributeError("region must have attribute od_demand (base 2D array).")

        q0_row = region.od_demand[0, :]  # shape (N,)
        if np.isnan(q0_row).any():
            raise ValueError(f"Found NaN in region.od_demand[0,:] for region {region_id}. Initialization incomplete.")

        within = float(q0_row[i_idx])
        inter = float(q0_row.sum() - q0_row[i_idx])

        demand_ii.append(within)
        demand_ij.append(inter)

    c_wr = (np.mean(demand_ii) * 0.4 * p1) if demand_ii else 0.0
    c_ir = (np.mean(demand_ij) * 0.15 *  p2) if demand_ij else 0.0

    print('c_wr:', c_wr)
    print('c_ir:', c_ir)
    return c_wr, c_ir


class CompoundOptionAnalysis: 
    def __init__(
        self,
        region_dict,
        T,
        c_wr,
        c_ir,
        discount_rate=0.01,
        n_paths=300,
        seed=20240908
    ):
        self.region_dict = region_dict
        self.T = T
        self.H = 0

        self.c_wr = c_wr
        self.c_ir = c_ir
        self.invest_sequence = []

        self.discount_rate = discount_rate
        self.n_paths = n_paths

        self.stopping_times = None
        self.option_values = None
        self.imm_values = None
        self.state_vars = None
        self.execution_times = None


        self.seed = seed

        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(self.n_paths)
        self.rng_paths = [np.random.default_rng(s) for s in child_seeds]

        self.rng = np.random.default_rng(self.seed)

        for reg in self.region_dict.values():
            reg.od_demand_paths = [
                reg.od_demand.copy() for _ in range(self.n_paths)
            ]
            for p in range(self.n_paths):
                reg.od_demand_paths[p][1:, :] = np.nan
       
        
        self.regression_models = {}
        
        self.region_num = len(self.region_dict)      # N
        self.kappa = 1.0                          # Experiment A sweep
        self.scale_mode = "nonstationary"          # "stationary" / "nonstationary"
        self.f_max = 2.0                             # cap for nonstationary f(I)
        self.scale_a = (self.f_max - 1.0) / np.log(1.0 + self.region_num)  # so f(N)=f_max
     
    def set_spillover_config(self, kappa=1.0, scale_mode="nonstationary", f_max=2.0):
        self.kappa = float(kappa)
        self.scale_mode = str(scale_mode)
        self.f_max = float(f_max)
        self.scale_a = (self.f_max - 1.0) / np.log(1.0 + self.region_num)

        for reg in self.region_dict.values():
            reg.kappa = self.kappa
            reg.scale_mode = self.scale_mode
            reg.f_max = self.f_max
            reg.scale_a = self.scale_a
        
    def reset_latent_paths(self):
        for reg in self.region_dict.values():
            for p in range(self.n_paths):
                reg.od_demand_paths[p][1:, :] = np.nan
        
        
    def _apply_diminishing(self, incremental_demand, mode="power", **kwargs):
        x = max(0.0, float(incremental_demand)) 

        if mode == "linear":
            return x 

        elif mode == "power":
            alpha = kwargs.get("alpha", 0.7)
            return x ** alpha

        elif mode == "log":
            beta = kwargs.get("beta", 1.0)
            return np.log1p(beta * x)

        elif mode == "logistic":
            K = kwargs.get("K", 1_000.0)
            gamma = kwargs.get("gamma", 0.01)
            return (K * x) / (1.0 + gamma * x)

        else:
            return x
    
    
    def calculate_immediate_payoff(
        self,
        t,
        invest_portfolio,
        distribution='gamma',
        diminish_mode="power",
        path_id=0,
        **diminish_kwargs
    ):


        if hasattr(self, "rng_paths") and self.rng_paths is not None:
            rng = self.rng_paths[path_id]
        else:
            rng = self.rng

        invested_regions = []
        imm = self.invest_sequence.index(invest_portfolio)
        for portfolio in self.invest_sequence[:imm]:
            invested_regions.extend(portfolio)

        Z = len(invested_regions)
        L_t = Z 

        demand_before_in = 0.0
        demand_before_out = 0.0

        if Z > 0:
            for origin_id in invested_regions:
                origin_region = self.region_dict[origin_id]

                demand_before_in += calculate_od_demand(
                    t, L_t, origin_region, origin_id,
                    distribution=distribution, rng=rng, path_id=path_id
                )

                for dest_id in invested_regions:
                    if dest_id != origin_id:
                        demand_before_out += calculate_od_demand(
                            t, L_t, origin_region, dest_id,
                            distribution=distribution, rng=rng, path_id=path_id
                        )

        new_invested = invested_regions + list(invest_portfolio)

        demand_after_in = 0.0
        demand_after_out = 0.0

        for origin_id in new_invested:
            origin_region = self.region_dict[origin_id]

            demand_after_in += calculate_od_demand(
                t, L_t, origin_region, origin_id,
                distribution=distribution, rng=rng, path_id=path_id
            )

            for dest_id in new_invested:
                if dest_id != origin_id:
                    demand_after_out += calculate_od_demand(
                        t, L_t, origin_region, dest_id,
                        distribution=distribution, rng=rng, path_id=path_id
                    )

        incremental_demand = (demand_after_in + demand_after_out) - (demand_before_in + demand_before_out)

        effective_demand = incremental_demand

        n_p = len(invest_portfolio)

        f_time = 1.0
        dynamic_c_wr = self.c_wr * f_time
        dynamic_c_ir = self.c_ir * f_time

        cost = n_p * dynamic_c_wr + 2.0 * dynamic_c_ir * (n_p * Z + n_p * (n_p - 1) / 2.0)

        payoff = effective_demand - cost
        return payoff, incremental_demand

    
    def create_basis_functions(self, x, degree=3):
        hermite_basis = [np.ones_like(x), x]

        for d in range(2, degree + 1):
            hermite_basis.append(x * hermite_basis[-1] - (d - 1) * hermite_basis[-2])

        return np.stack(hermite_basis).T
    
    def estimate_continuation_value(self, t, h, state_variables, future_cashflows):

        X = self.create_basis_functions(state_variables)
        
        model = LinearRegression().fit(X, future_cashflows)
        continuation_values = model.predict(X)

        self.regression_models[(t, h, 'continuation')] = model
        
        return continuation_values
    
    def estimate_next_option_value(self, t, h, state_variables, next_option_values):

        X = self.create_basis_functions(state_variables)
        
        model = LinearRegression().fit(X, next_option_values)
        expected_values = model.predict(X)
        
        self.regression_models[(t, h, 'next_option')] = model
        
        return expected_values

    def sequence_valuation(self, invest_sequence, distribution='gamma'):

        self.invest_sequence = invest_sequence
        self.H = len(invest_sequence)
        self.stopping_times = np.full((self.H, self.n_paths), self.T, dtype=int)

        self.option_values = np.zeros((self.T + 1, self.H, self.n_paths))
        self.imm_values = np.zeros((self.T + 1, self.H, self.n_paths))
        self.state_vars = np.zeros((self.T + 1, self.H, self.n_paths))

        for h in range(self.H):
            invest_portfolio = self.invest_sequence[h]
            last_t = self.T - (self.H - h) + 1

            for p in range(self.n_paths):
                self.stopping_times[h, p] = last_t

                payoff, _ = self.calculate_immediate_payoff(
                    t=last_t,
                    invest_portfolio=invest_portfolio,
                    distribution=distribution,
                    diminish_mode="power",
                    alpha=0.7,
                    path_id=p 
                )

                self.option_values[last_t, h, p] = payoff
                self.imm_values[last_t, h, p] = payoff

                for m_t in range(last_t + 1, self.T + 1):
                    self.option_values[m_t, h, p] = 0.0
                    self.imm_values[m_t, h, p] = 0.0

        for t in range(self.T - 1, 0, -1):
            for h in range(self.H - 1, -1, -1):
                invest_portfolio = self.invest_sequence[h]

                immediate_payoffs = np.zeros(self.n_paths)

                for p in range(self.n_paths):
                    if t <= self.stopping_times[h, p]:
                        imm_p, state_p = self.calculate_immediate_payoff(
                            t=t,
                            invest_portfolio=invest_portfolio,
                            distribution=distribution,
                            diminish_mode="power",
                            alpha=0.7,
                            path_id=p 
                        )
                        immediate_payoffs[p] = imm_p
                        self.state_vars[t, h, p] = state_p
                        self.imm_values[t, h, p] = imm_p
                    else:
    
                        immediate_payoffs[p] = 0.0


                if h < self.H - 1:
                    next_option_values = self.option_values[t + 1, h + 1, :]
                    discounted_next_values = (1 + self.discount_rate) ** (-1) * next_option_values

                    next_option_value = self.estimate_next_option_value(
                        t=t,
                        h=h,
                        state_variables=self.state_vars[t, h + 1, :],
                        next_option_values=discounted_next_values
                    )

                    immediate_payoffs += next_option_value

                    discounted_cashflows = np.zeros(self.n_paths)
                    for p in range(self.n_paths):
                        for t_i in range(t + 1, self.T + 1):
                            discount_factor = (1 + self.discount_rate) ** (-(t_i - t))
                            sum_pi = 0.0
                            for h_i in range(h, self.H):
                                if t_i == self.stopping_times[h_i, p]:
                                    sum_pi += self.imm_values[t_i, h_i, p]
                            discounted_cashflows[p] += discount_factor * sum_pi

                    continuation_values = self.estimate_continuation_value(
                        t=t,
                        h=h,
                        state_variables=self.state_vars[t, h, :],
                        future_cashflows=discounted_cashflows
                    )
                else:
                    continuation_values = np.zeros(self.n_paths)

                for p in range(self.n_paths):
                    if t <= self.stopping_times[h, p]:
                        if immediate_payoffs[p] >= continuation_values[p]:
                            self.stopping_times[h, p] = t
                            self.option_values[t, h, p] = immediate_payoffs[p]

                
                            for m in range(h + 1, self.H):
                                self.stopping_times[m, p] = max(
                                    self.stopping_times[m, p],
                                    self.stopping_times[m - 1, p] + 1
                                )
                        else:
                            for m in range(h, self.H):
                                self.option_values[t, m, p] = (1 + self.discount_rate) ** (-1) * self.option_values[t + 1, m, p]

        for h in range(self.H):
            for p in range(self.n_paths):
                stop_time = self.stopping_times[h, p]
                discount_factor = (1 + self.discount_rate) ** (-stop_time)
                self.option_values[0, h, p] = discount_factor * self.option_values[stop_time, h, p]

        sequence_value = self.option_values[0, 0, :].mean()

        Exec_times = {h: int(np.median(self.stopping_times[h, :])) for h in range(self.H)}
        for h in range(1, self.H):
            Exec_times[h] = max(Exec_times[h], Exec_times[h-1] + 1)

        return sequence_value, Exec_times

    
    def rank_sequence(self, sequence_list, distribution = 'gamma'):
        results = []
        for invest_sequence in sequence_list:
            sequence_value, Exec_times = self.sequence_valuation(invest_sequence, distribution)
            results.append({
                'invest_sequence': invest_sequence,
                'value': sequence_value,
                'Exec_times': Exec_times
            })

        sorted_results = sorted(results, key=lambda x: x['value'], reverse=True)
        return sorted_results


    def Future_NPV(
        self,
        invest_sequence,
        exec_times,
        n_paths=None,              
        distribution='gamma',
        apply_diminishing=False,  
        diminish_mode="power",
        **diminish_kwargs
    ):
        if n_paths is None:
            n_paths = self.n_paths

        for reg in self.region_dict.values():
            for p in range(n_paths):
                reg.od_demand_paths[p][1:, :] = np.nan

        ss = np.random.SeedSequence(self.seed + 99991) 
        child_seeds = ss.spawn(n_paths)
        rng_paths = [np.random.default_rng(s) for s in child_seeds]

        def total_demand_on_set(t, S, L_t, path_id):
            if not S:
                return 0.0
            demand_in, demand_out = 0.0, 0.0
            rng = rng_paths[path_id]

            for origin_id in S:
                origin_region = self.region_dict[origin_id]
                demand_in += calculate_od_demand(
                    t, L_t, origin_region, origin_id,
                    distribution=distribution, rng=rng, path_id=path_id
                )
                for dest_id in S:
                    if dest_id != origin_id:
                        demand_out += calculate_od_demand(
                            t, L_t, origin_region, dest_id,
                            distribution=distribution, rng=rng, path_id=path_id
                        )
            return demand_in + demand_out

        total_npv_paths = np.zeros(n_paths)

        for p in range(n_paths):
            total_npv = 0.0
            Z = []

            for t in range(self.T + 1):
                todays_ports = [port for k, port in enumerate(invest_sequence) if exec_times[k] == t]
                if not todays_ports:
                    continue

                P_all, seen = [], set()
                for port in todays_ports:
                    for rid in port:
                        if rid not in seen:
                            seen.add(rid)
                            P_all.append(rid)

                L_t = len(Z)

                D_before = total_demand_on_set(t, Z, L_t, path_id=p)
                Z_after  = Z + P_all
                D_after  = total_demand_on_set(t, Z_after, L_t, path_id=p)

                incremental_demand = D_after - D_before

                if apply_diminishing:
                    inc_eff = self._apply_diminishing(incremental_demand, mode=diminish_mode, **diminish_kwargs)
                else:
                    inc_eff = incremental_demand

                H_cov = len(Z)
                n_p = len(P_all)

                f_time = 1.0
                cost_portfolio = n_p * (self.c_wr * f_time) + 2.0 * (self.c_ir * f_time) * (
                    n_p * H_cov + n_p * (n_p - 1) / 2.0
                )

                npv_t = inc_eff - cost_portfolio
                total_npv += (1 + self.discount_rate) ** (-t) * npv_t

                Z = Z_after

            total_npv_paths[p] = total_npv

        return float(total_npv_paths.mean())



    





