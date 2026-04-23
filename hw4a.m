%% Load data
data = readtable("California_newHIVcases_byyear");
% Convert dates
years = table2array(data(:,1));
infected = table2array(data(:,2));
%%
% Time in years since first observation

infected_data = infected;
infected_time = years;

% Plot raw data
figure;
scatter(infected_time, infected_data, 10, 'r', 'filled')
title("Fake Infection Data")
ylabel("Cases")

%% Initial conditions
initial_conditions = [29999990; 5000; 500; 0];

%% Fixed parameters
pi_val = 1/9000;
delta_val = 1/9000;

%% Initial guesses for parameters to fit
% [beta, xi, sigma, gamma]
beta0 = [0.3, 1/90, 1/3, 1/7];

% Bounds (equivalent to min/max in lmfit)
lb = [0.1, 1/270, 1/5, 1/21];
ub = [3,   1/30,  1,   1/5];

%% Define model for fitting
model_fun = @(b, t) model_wrapper(b, t, initial_conditions, pi_val, delta_val);

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
plot(infected_dates, predicted_cases, 'LineWidth', 2)
hold on
scatter(infected_dates, infected_data, 10, 'r', 'filled')
ylabel("Cases")
title("Data Fitting Result")
xticks(infected_dates(1:10:end))
xtickangle(70)
legend("Predicted Cases (model)", "True Cases (data)")
hold off

%% Display fitted parameters
beta_fit

function I_out = model_wrapper(b, t, IC, pi_val, delta_val)
    beta  = b(1);
    xi    = b(2);
    sigma = b(3);
    gamma = b(4);

    params = [pi_val, xi, beta, delta_val, sigma, gamma];

    [t_sol, sol] = ode45(@(t,y) disease_dynamics(t,y,params), [t(1), t(end)], IC);

    I_out = interp1(t_sol, sol(:,3), t);
end


function ddt = disease_dynamics(t, states, params)
    S = states(1);
    E = states(2);
    I = states(3);
    R = states(4);

    N = sum(states);

    pi    = params(1);
    xi    = params(2);
    beta  = params(3);
    delta = params(4);
    sigma = params(5);
    gamma = params(6);

    dSdt = pi + xi*R - beta*(S/N)*I - delta*S;
    dEdt = beta*(S/N)*I - (sigma + delta)*E;
    dIdt = sigma*E - (gamma + delta)*I;
    dRdt = gamma*I - (delta + xi)*R;

    ddt = [dSdt; dEdt; dIdt; dRdt];
end
