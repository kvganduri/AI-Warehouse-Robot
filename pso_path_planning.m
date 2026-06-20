% =========================================================================
% Particle Swarm Optimization (PSO) for Mobile Robot Path Planning
% -------------------------------------------------------------------------
% Reconstructed from: Ganduri et al., "Adaptive Intelligence in Warehouse
% Robotics," J. Electrical Systems 20-3 (2024): 8215-8230, Section VI.C
% and "Algorithm 1: Particle Swarm Optimization For Mobile Robot."
%
% IMPORTANT - READ BEFORE USE:
% The pseudocode printed as "Algorithm 1" in the paper is not valid,
% runnable code. It contains undefined procedure arguments (p,s,c,s,g,l),
% a function (evaluate_fitness) called before it is defined, malformed
% syntax ("then" is not a MATLAB keyword), and incomplete parameter
% definitions ("Parameters (Area, dt, k_size, robots)" with no values).
%
% This script is NOT a transcription -- it is a working implementation
% that follows the algorithm's logic exactly as described in the paper's
% prose (Section VI.C): a swarm of particles representing candidate robot
% positions, updated via cognitive component (pull toward each particle's
% own best position) and social component (pull toward the swarm's global
% best), with fitness = Euclidean distance to the target, converging on
% a near-optimal path while avoiding a static obstacle (matching Fig. 13:
% start position in yellow, obstacle in green).
% =========================================================================

clear all;
close all;

%% ---- PSO and environment parameters ----
num_particles = 30;
dimensions    = 2;       % 2D warehouse floor (x, y)
max_iter      = 100;

w  = 0.7;   % inertia weight
c1 = 1.5;   % cognitive coefficient
c2 = 1.5;   % social coefficient

area_bounds = [-2 2; -2 2];   % [xmin xmax; ymin ymax] -- matches Fig.13 axes

start_point  = [-1.0, -1.5];   % robot start (yellow, Fig. 13a)
target_point = [ 1.2,  1.2];   % delivery / rack target
obstacle_pos = [ 0.0,  0.6];   % obstacle (green, Fig. 13a)
obstacle_radius = 0.35;

%% ---- Initialize swarm ----
positions = area_bounds(:,1)' + rand(num_particles, dimensions) .* ...
            (area_bounds(:,2)' - area_bounds(:,1)');
velocity  = zeros(num_particles, dimensions);

best_positions = positions;
best_values    = evaluate_fitness(positions, target_point, obstacle_pos, obstacle_radius);

[global_best_value, idx] = min(best_values);
global_best_position = best_positions(idx, :);

history_best_value = zeros(max_iter, 1);
swarm_history = cell(max_iter, 1);

%% ---- PSO main loop ----
for iter = 1:max_iter
    r1 = rand(num_particles, dimensions);
    r2 = rand(num_particles, dimensions);

    cognitive_component = c1 * r1 .* (best_positions - positions);
    social_component     = c2 * r2 .* (repmat(global_best_position, num_particles, 1) - positions);

    velocity  = w * velocity + cognitive_component + social_component;
    positions = positions + velocity;

    % keep particles within warehouse bounds
    positions(:,1) = min(max(positions(:,1), area_bounds(1,1)), area_bounds(1,2));
    positions(:,2) = min(max(positions(:,2), area_bounds(2,1)), area_bounds(2,2));

    current_values = evaluate_fitness(positions, target_point, obstacle_pos, obstacle_radius);

    improved = current_values < best_values;
    best_positions(improved, :) = positions(improved, :);
    best_values(improved) = current_values(improved);

    [current_global_best, idx] = min(best_values);
    if current_global_best < global_best_value
        global_best_value = current_global_best;
        global_best_position = best_positions(idx, :);
    end

    history_best_value(iter) = global_best_value;
    swarm_history{iter} = positions;
end

fprintf('PSO converged after %d iterations.\n', max_iter);
fprintf('Best path endpoint found: (%.3f, %.3f)\n', global_best_position(1), global_best_position(2));
fprintf('Distance to target: %.4f\n', global_best_value);

%% ---- Plot: initial swarm vs obstacle (Fig. 13a equivalent) ----
figure('Name', 'PSO Initial State', 'NumberTitle', 'off');
hold on; axis equal; grid on;
xlim(area_bounds(1,:)); ylim(area_bounds(2,:));
scatter(swarm_history{1}(:,1), swarm_history{1}(:,2), 25, [0.6 0.6 0.6], 'filled', ...
    'DisplayName', 'Particles (iter 1)');
plot(start_point(1), start_point(2), 'o', 'MarkerSize', 14, ...
    'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'DisplayName', 'Robot start');
viscircles(obstacle_pos, obstacle_radius, 'Color', 'g');
plot(obstacle_pos(1), obstacle_pos(2), 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
plot(target_point(1), target_point(2), 'p', 'MarkerSize', 16, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'Target');
title('Initial swarm: robot (yellow) and obstacle (green)');
xlabel('x (m)'); ylabel('y (m)');
legend('Location', 'bestoutside');

%% ---- Plot: final swarm / converged positions (Fig. 13b equivalent) ----
figure('Name', 'PSO Final State', 'NumberTitle', 'off');
hold on; axis equal; grid on;
xlim(area_bounds(1,:)); ylim(area_bounds(2,:));
scatter(positions(:,1), positions(:,2), 25, [0.2 0.4 0.9], 'filled', ...
    'DisplayName', 'Particles (final)');
plot(global_best_position(1), global_best_position(2), 'o', 'MarkerSize', 14, ...
    'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'DisplayName', 'Best path endpoint');
viscircles(obstacle_pos, obstacle_radius, 'Color', 'g');
plot(obstacle_pos(1), obstacle_pos(2), 'o', 'MarkerSize', 10, ...
    'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
plot(target_point(1), target_point(2), 'p', 'MarkerSize', 16, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'Target');
title('Converged swarm positions vs. obstacle');
xlabel('x (m)'); ylabel('y (m)');
legend('Location', 'bestoutside');

%% ---- Plot: convergence curve ----
figure('Name', 'PSO Convergence', 'NumberTitle', 'off');
plot(1:max_iter, history_best_value, 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Best fitness (distance to target)');
title('PSO Convergence Curve');
grid on;

%% ===================== Local function =====================
function fitness = evaluate_fitness(positions, target_point, obstacle_pos, obstacle_radius)
    % Fitness = Euclidean distance to target, as described in the paper
    % ("the fitness of each path is evaluated based on the Euclidean
    % distance to the target"), with an added penalty for entering the
    % obstacle's exclusion radius to implement the collision-avoidance
    % behavior referenced in Section VI.C.
    diffs = positions - repmat(target_point, size(positions,1), 1);
    distances = sqrt(sum(diffs.^2, 2));

    obst_diffs = positions - repmat(obstacle_pos, size(positions,1), 1);
    obst_dist  = sqrt(sum(obst_diffs.^2, 2));

    penalty = zeros(size(distances));
    inside_obstacle = obst_dist < obstacle_radius;
    penalty(inside_obstacle) = 10 * (obstacle_radius - obst_dist(inside_obstacle));

    fitness = distances + penalty;
end
