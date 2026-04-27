%% Load data
data = readtable("California_newHIVcases_byyear");
% Convert dates
years = table2array(data(:,1));
infected = table2array(data(:,2));
% Time in years since first observation

infected_data = infected;
infected_time = years;

% Plot raw data
figure;
scatter(infected_time, infected_data, 10, 'r', 'filled')
title("New HIV Infections in California")
ylabel("Year")

%% Initial conditions
initial_conditions = [0; 20000; 500; 400; 0];

%% Fixed parameters
pi_val = 376078.91;
mu_val = 0.009043;
sigma_val = 0.0885;
tau_I = 0.2;
tau_A = 0.25;
beta_val = 9.61*10^-6;
delta_I_val = 0.0158;
delta_A_val = 1/3;
A = [pi_val, mu_val, sigma_val, tau_I, tau_A, beta_val, delta_I_val, delta_A_val];
%% Initial guesses for parameters to fit
% [eta, epsilon, q]
beta0 = [0.5, 0.95, 0.037];

% Bounds (equivalent to min/max in lmfit)
lb = [0, 0, 0];
ub = [1, 1, 1];

%% Define model for fitting
model_fun = @(b, t) model_wrapper(b, t, initial_conditions, A);

%% Perform fitting
options = optimoptions('lsqcurvefit', ...
    'Display','iter', ...
    'MaxFunctionEvaluations', 200, ...
    'MaxIterations', 50);

beta_fit = lsqcurvefit(model_fun, beta0, infected_time, infected_data, lb, ub, options);

%% Get predicted values
predicted_cases = model_fun(beta_fit, infected_time);

%% Plot results
figure;
plot(infected_time, predicted_cases, 'LineWidth', 2)
hold on
scatter(infected_time, infected_data, 10, 'r', 'filled')
ylabel("Cases")
title("Data Fitting Result")
legend("Predicted Cases (model)", "True Cases (data)")
hold off

%% Display fitted parameters
beta_fit

function I_out = model_wrapper(b, t, IC, A)
    eta  = b(1);
    epsilon = b(2);
    q = b(3);
    params = [A, eta, epsilon, q];

    [t_sol, sol] = ode45(@(t,y) disease_dynamics(t,y,params), [t(1), t(end)], IC);

    I_out = interp1(t_sol, sol(:,3), t);
end


function ddt = disease_dynamics(t, states, params)
    Sl = states(1);
    Sh = states(2);
    I = states(3);
    A = states(4);
    T = states(5);

    N = sum(states);

    pi    = params(1);
    mu  = params(2);
    sigma = params(3);
    tau_I = params(4);
    tau_A = params(5);
    beta = params(6);
    delta_I = params(7);
    delta_A = params(8);
    eta = params(9);
    epsilon = params(10);
    q = params(11);

    dSldt = pi*(1-q)-beta*((eta*I+A)/N)*Sl-mu*Sl;
    dShdt = pi*q-beta*(1-epsilon)*((eta*I+A)/N)*Sh-mu*Sh;
    dIdt = beta*((eta*I+A)/N)*(Sl+(1-epsilon)*Sh)-sigma*I-tau_I*I-mu*I-delta_I*I;
    dAdt = sigma*I-tau_A*A-mu*A-delta_A*A;
    dTdt = tau_I*I+tau_A*A-mu*T;

    ddt = [dSldt; dShdt; dIdt; dAdt ; dTdt];
end
