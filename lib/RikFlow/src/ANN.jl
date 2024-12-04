
"""
    _normalise(x; normalization=:normal, dims=ndims(x), ϵ=1e-6)

Normalize the input array `x` using the specified normalization method.

# Arguments
- `x`: The input array to be normalized.
- `normalization`: The normalization method to use. Can be `:normal` for standard normalization or `:minmax` for min-max normalization. Default is `:normal`.
- `dims`: The dimensions over which to compute the normalization statistics. Default is `ndims(x)`.
- `ϵ`: A small value added to the standard deviation to avoid division by zero. Default is `1e-6`.

# Returns
- A tuple containing the normalized array and a named tuple with the computed mean (`mu`) and standard deviation (`sigma`).

# Normalization Methods
- `:normal`: Standard normalization where the mean (`mu`) and standard deviation (`sigma`) are computed along the specified dimensions.
- `:minmax`: Min-max normalization where the minimum and maximum values are computed along the specified dimensions, and the mean (`mu`) is set to the midpoint and the standard deviation (`sigma`) to half the range.
"""
function _normalise(x; normalization= :normal, dims=ndims(x), ϵ=1e-6)
    if normalization == :normal
        ϵ = convert(eltype(x), ϵ)
        mu = mean(x, dims=dims)
        sigma = std(x, dims=dims, corrected=false).+ ϵ
        
    elseif normalization == :minmax
        min = minimum(x, dims=dims)
        max = maximum(x, dims=dims)
        mu = 0.5(min+max)
        sigma = 0.5(max-min)
    end
    return (x .- mu) ./ (sigma), (;mu,sigma)
end

function scale_input(x::AbstractArray, scaling)
    return (x .- scaling.mu) ./ scaling.sigma
end

function scale_output(x::AbstractArray, scaling)
    return x .* scaling.sigma .+ scaling.mu
end

function setup_ANN(n_qois, hist_len, out_size, hidden_size, n_layers, rng; activation_function = leakyrelu)
    model_dict = (; n_qois, hist_len, out_size, hidden_size, n_layers, activation_function)
    in_size = n_qois *(1+ hist_len)
    if activation_function == leakyrelu
        last_activation = identity
    elseif activation_function == tanh
        last_activation = tanh
    end
    model = Chain(Dense(in_size => hidden_size, activation_function), 
                (Dense(hidden_size => hidden_size, activation_function) for _ in 1:n_layers-2)..., 
                Dense(hidden_size => out_size, last_activation))
    # Parameter and State Variables
    ps, st = Lux.setup(rng, model) |> dev
    return model, ps, st, model_dict
end

function load_ANN(file_name)
    (model_dict, parameters, states, scaling) = load(file_name, "model_dict", "parameters", "states", "scaling")
    (; hidden_size, n_layers, n_qois, hist_len, out_size, activation_function) = model_dict
    in_size = n_qois *(1+ hist_len)
    if activation_function == leakyrelu
        last_activation = identity
    elseif activation_function == tanh
        last_activation = tanh
    end
    model = Chain(Dense(in_size => hidden_size, activation_function), 
                (Dense(hidden_size => hidden_size, activation_function) for _ in 1:n_layers-2)..., 
                Dense(hidden_size => out_size, last_activation))
    ps = parameters |> dev
    st = states |> dev
    scaling = scaling |> dev
    return model, ps, st, scaling
end

function create_dataloader(data_in, data_out , batchsize, rng; normalization = :normal)
    data_in_scaled, in_scaling = _normalise(data_in; normalization)
    data_out_scaled, out_scaling = _normalise(data_out; normalization)
    (in_train, out_train),(in_val, out_val) = splitobs((data_in_scaled, data_out_scaled); at = 0.9, shuffle = true) # can not pass rng!
    val_loader = DataLoader(collect.((in_val, out_val)); batchsize, rng) |> dev
    train_loader = DataLoader(collect.((in_train, out_train)); batchsize, shuffle=true, rng) |> dev
    loaders = (;train_loader, val_loader)
    return loaders, (;in_scaling, out_scaling)
end

function train_model(tstate::Training.TrainState, dataloaders, loss_function, epochs, vjp = AutoZygote())
    eta = tstate.optimizer.eta
    losses = (train=[], val=[])
    for epoch in 1:epochs
        test_loss = 0
        i=0
        for (x,y) in dataloaders.train_loader
            _, batch_loss, _, tstate = Training.single_train_step!(vjp, loss_function, (x,y), tstate)
            test_loss += batch_loss
            i+=1
        end
        test_loss /= i
        push!(losses.train, test_loss)
        if epoch % 100 ==1 || epoch == epochs
            st_ = Lux.testmode(tstate.states)
            val_loss = 0
            i=0
            for (x,y) in dataloaders.val_loader
                y_pred, st_ = tstate.model(x, tstate.parameters, st_)
                val_loss += loss_function(y_pred, y)
                i+=1
            end
            val_loss /= i
            push!(losses.val, val_loss)
            @info epoch test_loss val_loss
        end
        if epoch % 100 == 0
            if losses.train[end] > losses.train[end-99] && losses.val[end] > losses.val[end-1]
                eta *= 0.1
                @info "Reducing learning rate to $eta"
                Optimisers.adjust!(tstate.optimizer_state; eta)
            end
        end
    end
    return tstate, losses
end
export scale_input, scale_output, setup_ANN, load_ANN, create_dataloader, train_model