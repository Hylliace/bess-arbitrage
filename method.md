# Method

The problem is split in two. A forecast gives the prices, then an optimizer picks the orders under the battery constraints.

## The battery

Capacity $C = 2$ MWh, power $P = 1$ MW, one step is one hour. The round trip efficiency is 90%, split evenly between charging and discharging: $\eta = \sqrt{0.9}$.

At hour $t$, $E_t$ is the stored energy, $p_t$ the price in €/MWh and $q_t$ the energy exchanged with the grid, positive when buying and negative when selling. With $q_t^+ = \max(q_t, 0)$ and $q_t^- = \max(-q_t, 0)$:

$$
E_{t+1} = E_t + \eta q_t^+ - \frac{q_t^-}{\eta}, \qquad 0 \le E_t \le C, \qquad |q_t| \le P.
$$

The wear cost $c$ is counted per MWh taken out of the battery, so the gain of an hour and over $T$ hours are

$$
g_t = -p_t q_t - c \frac{q_t^-}{\eta}, \qquad G = \sum_{t=0}^{T-1} g_t.
$$

The battery starts and ends each period empty, $E_0 = E_T = 0$, but it does not have to be empty every night. Forcing that would rule out buying late in the evening to sell the next morning.

Cash can go negative at the start, since the battery buys before it sells, and I do not put a limit on that.

### Wear cost

Without a wear cost, the battery would cycle even for a few euros, which makes no sense for a real one. I use a linear cost. A cell replacement at $K$ €/kWh spread over $N$ full cycles gives

$$
c = \frac{1000 K}{N}.
$$

With $N = 4000$, $K = 100$ €/kWh gives 25 €/MWh and $K = 160$ €/kWh gives 40 €/MWh. These are rough guesses based on the [BloombergNEF 2025 survey](https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-despite-rising-metal-prices-bloombergnef/), [NREL ATB 2024](https://atb.nrel.gov/electricity/2024/commercial_battery_storage) and the datasheet of an [EVE MB31](https://www.evemall.eu/power-battery/prismatic-lfp-cell/mb31) LFP cell. The residual value is taken as zero and the installation is not counted.

## When decisions are made

To deliver on day $d$, orders are sent at 10am on day $d-1$. At that point the day-ahead prices of $d-1$ are known (they were set on $d-2$) but not those of $d$. I forecast $d$ and $d+1$, plan both days, and only send the orders of $d$. The second day is only there so that the plan can decide to keep energy for tomorrow.

The orders are assumed to be fully accepted at the market price, and the battery is too small to move that price. There is no order book and no intraday market.

## Forecasts

The weekly forecast takes the same local hour seven days before:

$$
\hat p_{d+k,h} = p_{d+k-7,h}, \qquad k \in \{0, 1\}.
$$

For ridge, each example $i$ is an hour $h_i$ of day $d_i + k_i$, where $d_i$ is the first day of the plan and $k_i \in \{0, 1\}$. The first three inputs are past prices,

$$
x_{i,1} = p_{d_i-1,h_i}, \qquad x_{i,2} = p_{d_i-2,h_i}, \qquad x_{i,3} = p_{d_i+k_i-7,h_i},
$$

then come one-hot columns for the hour (24) and for the day of the week of the delivery (7), and $k_i$ itself, so 35 inputs in total.

The second day gets the same two recent prices as the first one, since the prices of the first day are not known yet.

Each column is standardized with the mean $\mu_j$ and standard deviation $\sigma_j$ of the training set, $z_{ij} = (x_{ij} - \mu_j)/\sigma_j$ (with $\sigma_j$ replaced by 1 for constant columns). The forecast is $\hat p_i = b + z_i^\top \beta$ with

$$
\min_{b, \beta} \sum_i (p_i - b - z_i^\top \beta)^2 + \alpha \|\beta\|^2,
$$

so $b$ is the mean price and $(Z^\top Z + \alpha I)\beta = Z^\top(p - b)$.

To pick $\alpha$, I tried 0.1, 1, 10, 100 and 1000 on three windows of 2022 (July-August, September-October, November-December), each time training only on what came before the window, and kept the value with the lowest mean absolute error, which was $\alpha = 0.1$. The model is then trained on all of 2022-2023.

The weekend variant adds three columns, equal to the three past prices on Saturdays and Sundays and to 0 on the other days. The same selection on 2022 gives $\alpha = 100$ for it.

## From prices to orders

The stored energy is put on a grid $\mathcal E = \{0, \delta, 2\delta, \ldots, C\}$ with $\delta = C/100$. Let $F(E, q)$ be the stored energy after exchanging $q$, from the battery equation above. The allowed exchanges are

$$
\mathcal A(E) = \{ q : |q| \le P, \ F(E, q) \in \mathcal E \}.
$$

For a plan over $H$ hours with forecast prices $\hat p_t$, going backwards from the end:

$$
V_t(E) = \max_{q \in \mathcal A(E)} \left[ -\hat p_t q - c \frac{\max(-q, 0)}{\eta} + V_{t+1}(F(E, q)) \right],
$$

with $V_H(0) = 0$ and $V_H(E) = -\infty$ otherwise. Then I go forward from the current stored energy and read the best $q$ at each step. There is only one signed exchange per hour, so the battery cannot charge and discharge at the same time.

On the last day of a period the plan only covers that day, so the battery does end empty.

The perfect forecast is the same computation with the real prices over the whole six months. It is the best result on the grid, not over all possible stored energies, but I tried 50, 100, 200 and 400 levels and with 100 the gain is within 0.13% of the one with 400 on the months I checked.

## Evaluation

The mean absolute error is computed on the hours that are actually delivered (the first day of each plan):

$$
\mathrm{MAE} = \frac{1}{N} \sum_t |\hat p_t - p_t|.
$$

The data keeps every delivery hour, including the 23 and 25 hour days when the clocks change. For the forecast inputs only, the repeated hour in October is averaged and the missing hour in March is the mean of its neighbours.

Periods: 2022-2023 to build the models, January-June 2024 to make the last choices (the weekend variant was dropped there), July-December 2024 as the final test, and January-June 2025 as a second check without retraining.
