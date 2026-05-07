%Group A: New HIV Cases in %California
%By Sofia Fontana and Connor Heft
%HW5. Part A Model with PrEP
%Loading Data
%% Load data
data = readtable("California_newHIVcases_byyear.txt");
% Convert dates
years = table2array(data(:,1));
infected = table2array(data(:,2));
% Time in years since first observation

infected_data = infected;
infected_time = years;

%% Initial conditions, same for all parts
initial_conditions = [29196876; 1121790; 5000; 3000; 100000]; % Sl, Sh, I, A, T

%% Fixed parameters
pi_val = 275537;
q_val = 0.037;
eta_val = 0.7;
mu_val = 0.009043;
epsilon_val = 0.99;
sigma_val = 0.0885;
tau_I = 0.6;
tau_A = 1.2;
cp_val = 1/3;

% Part a. Model without PrEP
A = [pi_val, q_val, eta_val, mu_val, epsilon_val, sigma_val, tau_I, tau_A, cp_val];

%% Function

function ddt = disease_dynamics(t, states, params)
    Sl = states(1);
    Sh = states(2);
    I = states(3);
    A = states(4);
    T = states(5);

    N = sum(states);

    pi = params(1);
    q  = params(2);
    eta = params(3);
    mu = params(4);
    epsilon = params(5);
    sigma = params(6);
    tau_I = params(7);
    tau_A = params(8);
    cp_val = params(9);
    beta = params(10);
    delta_I = params(11);
    deltA = params(12);

    dSldt = pi*(1-q)-beta*((eta*I+A)/N)*Sl-mu*Sl;
    dShdt = pi*q-beta*(1-epsilon*cp_val)*((eta*I+A)/N)*Sh-mu*Sh;
    dIdt = beta*((eta*I+A)/N)*(Sl+(1-epsilon*cp_val)*Sh)-sigma*I-tau_I*I-mu*I-delta_I*I;
    dAdt = sigma*I-tau_A*A-mu*A-deltA*A;
    dTdt = tau_I*I+tau_A*A-mu*T;

    ddt = [dSldt; dShdt; dIdt; dAdt ; dTdt];
end

%% HW6
infected_time10 = [infected_time; 2018; 2019; 2020; 2021];
beta = 1;
delta_I = 0.034185;
delta_A = 0.84026;
[t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
Sl = sol(:,1);
Sh = sol(:,2);
I  = sol(:,3);
A_state  = sol(:,4);   % avoid name conflict
N  = sum(sol,2);
death = delta_I*I + delta_A*A_state;
death = death/365;
incidence = (beta*((eta_val*I + A_state)./N).*(Sl + (1-epsilon_val*cp_val)*Sh))/365;



% Part a.i
figure;
plot(infected_time10, incidence, 'LineWidth', 2)
hold on
ylabel("New Daily Cases")
xlabel("Year")
title("Daily Incidences")
hold off

% Part a.iii
figure;
plot(infected_time10, death, 'LineWidth', 2)
hold on
ylabel("New Death Rate")
xlabel("Year")
title("Daily Death Rate")
hold off

% Part b
figure;
for i = 0:2
    eta_val1 = 0.5+(i*0.25);
    A = [pi_val, q_val, eta_val1, mu_val, epsilon_val, sigma_val, tau_I, tau_A,cp_val];
    [t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
    Sl = sol(:,1);
    Sh = sol(:,2);
    I  = sol(:,3);
    A_state  = sol(:,4);
    N  = sum(sol,2);
    incidence = (beta*((eta_val*I + A_state)./N).*(Sl + (1-epsilon_val*cp_val)*Sh))/365;

    plot(infected_time10, incidence, 'LineWidth', 2); hold on
    ylabel("Daily Incidence")
    xlabel("Year")
    title("Effect of Transmission")
end
    legend("eta = 0.5", "eta = 0.75","eta = 1", 'Location', 'Best')
hold off

figure;
for i = 0:2
    eta_val2 = 0.5+(i*0.25);
    A = [pi_val, q_val, eta_val2, mu_val, epsilon_val, sigma_val, tau_I, tau_A, cp_val];
    [t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
    I  = sol(:,3);
    A_state  = sol(:,4);
    death = delta_I*I + delta_A*A_state;
    death = death/365;

    plot(infected_time10, death, 'LineWidth', 2); hold on
    ylabel("Daily Death")
    xlabel("Year")
    title("Effect of Transmission")
end
    legend("eta = 0.5", "eta = 0.75","eta = 1", 'Location', 'Best')
hold off

% part c
figure;
for i = 0:4
    q_val1 = 0+(i*0.25);
    A = [pi_val, q_val1, eta_val, mu_val, epsilon_val, sigma_val, tau_I, tau_A, cp_val];
    [t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
    Sl = sol(:,1);
    Sh = sol(:,2);
    I  = sol(:,3);
    A_state  = sol(:,4);
    N  = sum(sol,2);
    incidence = (beta*((eta_val*I + A_state)./N).*(Sl + (1-epsilon_val*cp_val)*Sh))/365;

    plot(infected_time10, incidence, 'LineWidth', 2); hold on
    ylabel("Daily Incidence")
    xlabel("Year")
    title("Effect of risk structure")
end
    legend("q = 0","q=0.25","q = 0.5", "q = 0.75","q = 1", 'Location', 'Best')
hold off

%part e
figure;
for i = 1:4
    if i == 1
        newtau_I = 0.05;
        newtau_A = newtau_I;
    elseif i == 2
        newtau_I = 0.1;
        newtau_A = newtau_I;
    elseif i == 3
        newtau_I = 0.2;
        newtau_A = newtau_I;
    elseif i==4
        newtau_I = 1;
        newtau_A = newtau_I;
    end
    A = [pi_val, q_val, eta_val, mu_val, epsilon_val, sigma_val, newtau_I, newtau_A, cp_val];
    [t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
    Sl = sol(:,1);
    Sh = sol(:,2);
    I  = sol(:,3);
    A_state  = sol(:,4);
    N  = sum(sol,2);
    incidence = (beta*((eta_val*I + A_state)./N).*(Sl + (1-epsilon_val*cp_val)*Sh))/365;

    plot(infected_time10, incidence, 'LineWidth', 2); hold on
    ylabel("Daily Incidence")
    xlabel("Year")
    title("Effect of ART (Incidences)")
end
    legend("tau = 0.05","tau=0.1","tau=0.2", "tau=1", 'Location', 'Best')
hold off

figure;
for i = 1:4
    if i == 1
        newtau_I = 0.05;
        newtau_A = newtau_I;
    elseif i == 2
        newtau_I = 0.1;
        newtau_A = newtau_I;
    elseif i == 3
        newtau_I = 0.2;
        newtau_A = newtau_I;
    elseif i==4
        newtau_I = 1;
        newtau_A = newtau_I;
    end
    A = [pi_val, q_val, eta_val, mu_val, epsilon_val, sigma_val, newtau_I, newtau_A, cp_val];
    [t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
    Sl = sol(:,1);
    Sh = sol(:,2);
    I  = sol(:,3);
    A_state  = sol(:,4);
    N  = sum(sol,2);
    death = delta_I*I + delta_A*A_state;
    death = death/365;

    plot(infected_time10, death, 'LineWidth', 2); hold on
    ylabel("Daily Death")
    xlabel("Year")
    title("Effect of ART (Deaths)")
    cumdeath = trapz(infected_time10, death)
end
    legend("tau = 0.05","tau=0.1","tau=0.2", "tau=1", 'Location', 'Best')
hold off

%part d
figure;
for i = 1:4
    if i == 1
        eff = 0;
    elseif i == 2
        eff = 0.5;
    elseif i == 3
        eff = 0.75;
    elseif i==4
        eff = 0.95;
    end
    A = [pi_val, q_val, eta_val, mu_val, eff, sigma_val, tau_I, tau_A, cp_val];
    [t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
    Sl = sol(:,1);
    Sh = sol(:,2);
    I  = sol(:,3);
    A_state  = sol(:,4);
    N  = sum(sol,2);
    incidence = (beta*((eta_val*I + A_state)./N).*(Sl + (1-eff*cp_val)*Sh))/365;

    plot(infected_time10, incidence, 'LineWidth', 2); hold on
    ylabel("Daily Incidence")
    xlabel("Year")
    title("Effect of PrEP")
end
    legend("epsilon = 0","epsilon=0.5","epsilon=0.75", "epsilon=0.95", 'Location', 'Best')
hold off

figure;
for i = 1:4
    if i == 1
        con = 0;
    elseif i == 2
        con = 0.5;
    elseif i == 3
        con = 0.75;
    elseif i==4
        con = 0.95;
    end
    A = [pi_val, q_val, eta_val, mu_val, eff, sigma_val, tau_I, tau_A, cp_val];
    [t_sol, sol] = ode45(@(t,y)disease_dynamics(t,y,[A, beta, delta_I, delta_A]), infected_time10, initial_conditions);
    Sl = sol(:,1);
    Sh = sol(:,2);
    I  = sol(:,3);
    A_state  = sol(:,4);
    N  = sum(sol,2);
    incidence = (beta*((eta_val*I + A_state)./N).*(Sl + (1-epsilon_val*con)*Sh))/365;

    plot(infected_time10, incidence, 'LineWidth', 2); hold on
    ylabel("Daily Incidence")
    xlabel("Year")
    title("Effect of PrEP, % pop. on PrEP")
end
    legend("pop = 0","pop=0.5","pop=0.75", "pop=0.95", 'Location', 'Best')
hold off

