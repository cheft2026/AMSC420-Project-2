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
initial_conditions = [40049097.68; 1538750.378; 5000; 3000; 2000];

%% Fixed parameters (no PrEP)
pi_val = 376078.91;
mu_val = 0.009043;
sigma_val = 0.0885;
tau_I = 0;
tau_A = 0;
epsilon = 0;
eta = 0.7;
q = 0.037;

A = [pi_val, mu_val, sigma_val, tau_I, tau_A, epsilon, eta, q];
%% Initial guesses for parameters to fit
% [beta, deltaI, deltaA]
beta0 = [9.61*10^-6, 0.01598, 0.47];

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

[beta_fit,resnorm,resid,exitflag,output,lambda,J] = lsqcurvefit(model_fun, beta0, infected_time, infected_data, lb, ub, options);

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

function new_cases = model_wrapper(b, t, IC, A)
    beta  = b(1);
    delta_I = b(2);
    delta_A = b(3);
    params = [A, beta, delta_I, delta_A];

    [t_sol, sol] = ode45(@(t,y) disease_dynamics(t,y,params), [t(1), t(end)], IC);

    % Extract states
    Sl = sol(:,1);
    Sh = sol(:,2);
    I  = sol(:,3);
    A_state  = sol(:,4);   % avoid name conflict
    N  = sum(sol,2);

    eta = params(7);
    epsilon = params(6);
    beta = params(9);

    incidence = beta*((eta*I + A_state)./N).*(Sl + (1-epsilon)*Sh);

        % Return values at data time points
    new_cases = interp1(t_sol, incidence, t, 'linear', 'extrap');

%     new_cases = zeros(size(t));
% 
%     for i = 2:length(t)
%         % indices for this year interval
%         idx = (t_sol >= t(i-1)) & (t_sol <= t(i));
% 
%         if sum(idx) > 1
%             new_cases(i) = trapz(t_sol(idx), incidence(idx));
%         else
%             new_cases(i) = 0;
%         end
%     end
%     % handle first point (no previous year)
%     new_cases(1) = new_cases(2);
end
function ddt = disease_dynamics(t, states, params)
    Sl = states(1);
    Sh = states(2);
    I = states(3);
    A = states(4);
    T = states(5);

    N = sum(states);

    pi = params(1);
    mu = params(2);
    sigma = params(3);
    tau_I = params(4);
    tau_A = params(5);
    epsilon = params(6);
    eta = params(7);
    q = params(8);
    beta = params(9);
    delta_I = params(10);
    delta_A = params(11);

    dSldt = pi*(1-q)-beta*((eta*I+A)/N)*Sl-mu*Sl;
    dShdt = pi*q-beta*(1-epsilon)*((eta*I+A)/N)*Sh-mu*Sh;
    dIdt = beta*((eta*I+A)/N)*(Sl+(1-epsilon)*Sh)-sigma*I-tau_I*I-mu*I-delta_I*I;
    dAdt = sigma*I-tau_A*A-mu*A-delta_A*A;
    dTdt = tau_I*I+tau_A*A-mu*T;

    ddt = [dSldt; dShdt; dIdt; dAdt ; dTdt];
end
%%
beta = beta_fit(:,1);
delta_I = beta_fit(:,2);
delta_A = beta_fit(:,3);

R0 = (beta*eta*(tau_A+mu_val+delta_A)+sigma_val*beta)/((sigma_val+tau_I+mu_val+delta_I)*(tau_A+mu_val+delta_A))
RV = ((beta*eta*(tau_A+mu_val+delta_A)+sigma_val*beta)*(q-epsilon*q))/((sigma_val+tau_I+mu_val+delta_I)*(tau_A+mu_val+delta_A))

hit = (1-(1/R0))

ci = nlparci(beta_fit,resid,'jacobian', J)