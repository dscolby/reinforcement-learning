using Gadfly
using Revise
using StatsBase

"""Struct to store the necessary componeents of a Markov Decision Process"""
struct MDP
    states::Tuple
    actions::Tuple
    rewards::Dict
    transition_probabilities::Dict
    num_states::Int64
    num_actions::Int64
    γ::Float64
    values::Dict
    π::Vector{Float64}
end

"""
    MDP(states, actions, rewards, transitiion_probabilities)

    Create an MDP struct to solve
"""
function MDP(states::Tuple, actions::Tuple, rewards::Dict, transition_probabilities::Dict, 
    γ::Float64=0.9)

    # Initialize values to zero    
    values = Dict(state => 0.0 for state in states)

        MDP(states, actions, rewards, transition_probabilities, size(states, 1), 
            size(actions, 1), γ, values, zeros(size(states, 1)))
end

function gettransitionprobability(mdp::MDP, state, action)
    # Get the next state based on its probability given the current state and action
    possible_states = collect(keys(mdp.transition_probabilities[action][state]))
    state_probs = ProbabilityWeights(collect(values(mdp.transition_probabilities[action][state])))
    next_state = sample(possible_states, state_probs)

    return mdp.transition_probabilities[action][state][next_state]
end

function bellmanupdate(mdp::MDP, action_values::Dict, state, action)

    # Calculate P(s'|s, a)
    transition_probability = gettransitionprobability(mdp, state, action)

    # Get the immediate reward for that state
    reward = mdp.rewards[state][action]

    # Immediate reward plus discounted sum of values for next possible states
    action_values[action] = float(reward) + (mdp.γ*sum(transition_probability .* values(mdp.values)))
    
    return action_values
end

function estimateV!(mdp::MDP)
    Δ = Vector{Float64}(undef, mdp.num_states)
    for (i, state) in enumerate(mdp.states)
        temp = deepcopy(mdp.values[state]) 
        action_values = Dict(action => 0.0 for action in mdp.actions)
        
        # Collect the rewards for taking each action at each state
        for action in mdp.actions

            # Get updated values for each action
            action_values = bellmanupdate(mdp, action_values, state, action)

        # Update the MDP value with the highest temp value caused by the best action
        mdp.values[state] = maximum(collect(values(action_values)))
        Δ[i] = abs(temp - mdp.values[state])
        end
    end
    return maximum(Δ)
end

function getpolicy(mdp::MDP)
    policy = Dict()
    for state in mdp.states in 
        action_values = Dict()
        
        for action in mdp.actions
           action_values = bellmanupdate(mdp, action_values, state, action) 
        end
        policy[state] = findmax(action_values)[2]
    end
    return policy
end

function solve!(mdp::MDP, tolerance::Float64=1e-5)
    iteration = 0
    Δ = estimateV!(mdp)

    while Δ > tolerance
        iteration += 1
        Δ = estimateV!(mdp)

        if Δ < tolerance
            break
        end
        println("Change in state value: " * string(Δ))
    end
    return getpolicy(mdp), iteration
end

states = Tuple(i for i in 0:3)
actions = tuple("Left", "Right")
rewards = Dict(0 => Dict("Left" => -1, "Right" => -1), 
           1 => Dict("Left" => -1, "Right" => -1), 
           2 => Dict("Left" => -1, "Right" => 5),
           3 => Dict("Left" => -1, "Right" => 0))
transition_probabilities = Dict("Left" => Dict(0 => Dict(0 => 0.8, 1 => 0.2, 2 => 0, 3 => 0), 
                                               1 => Dict(0 => 0.8, 1 => 0, 2 => 0.2, 3 => 0), 
                                               2 => Dict(0 => 0, 1 => 0.8, 2 => 0, 3 => 0.2), 
                                               3 => Dict(0 => 0, 1 => 0, 2 => 0, 3 => 0)), 
                                "Right" => Dict(0 => Dict(0 => 0.2, 1 => 0.8, 2 => 0, 3 => 0), 
                                                1 => Dict(0 => 0.2, 1 => 0, 2 => 0.8, 3 => 0), 
                                                2 => Dict(0 => 0, 1 => 0.2, 2 => 0, 3 => 0.8), 
                                                3 => Dict(0 => 0, 1 => 0, 2 => 0, 3 => 0)))

simple_mdp = MDP(states, actions, rewards, transition_probabilities)
solve!(simple_mdp)
