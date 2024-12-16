# Neural closure models

!!! note "`NeuralClosure`"
    These features are experimental, and require cloning
    IncompressibleNavierStokes from GitHub:
    ```sh
    git clone https://github.com/agdestein/IncompressibleNavierStokes.jl
    cd IncompressibleNavierStokes/lib/NeuralClosure
    ```

Large eddy simulation, a closure model is required. With
IncompressibleNavierStokes, a neural closure model can be trained on filtered
DNS data. The discrete DNS equations are given by

```math
\begin{split}
M u & = 0, \\
\frac{\mathrm{d} u}{\mathrm{d} t} & = F(u) - G p.
\end{split}
```

Applying a spatial filter ``\Phi``, the extracted large scale components ``\bar{u} = \Phi u`` are governed by the equation

```math
\begin{split}
M \bar{u} & = 0, \\
\frac{\mathrm{d} \bar{u}}{\mathrm{d} t} & = F(\bar{u}) + c - G \bar{p},
\end{split}
```

where the discretizations ``M``, ``F``, and ``G`` are adapted to the size of
their inputs and ``c = \overline{F(u)} - F(\bar{u})`` is a commutator error. We
here assumed that ``M`` and ``\Phi`` commute, which is the case for face
averaging filters. Replacing ``c`` with a parameterized closure model
``m(\bar{u}, \theta) \approx c`` gives the LES equations for the approximate
large scale velocity ``\bar{v} \approx \bar{u}``

```math
\begin{split}
M \bar{v} & = 0, \\
\frac{\mathrm{d} \bar{v}}{\mathrm{d} t} & = F(\bar{v}) + m(\bar{v}, \theta) - G \bar{q}.
\end{split}
```

## NeuralClosure module

IncompressibleNavierStokes provides a NeuralClosure module.

```@autodocs
Modules = [NeuralClosure]
Pages = ["NeuralClosure.jl", "closure.jl"]
```

## Filters

The following filters are available:

```@autodocs
Modules = [NeuralClosure]
Pages = ["filter.jl"]
```

## Training

To improve the model parameters, we exploit exact filtered DNS data ``\bar{u}``
and exact commutator errors ``c`` obtained through DNS. The model is trained by
minimizing the a priori loss function

```math
L^\text{prior}(\theta) = \| m(\bar{u}, \theta) - c \|^2,
```

or the a posteriori loss function

```math
L^\text{post}(\theta) = \| \bar{v}_\theta - \bar{u} \|^2,
```

where ``\bar{v}_\theta`` is the solution to the LES equation for the given
parameters ``\theta``. The prior loss is easy to evaluate and easy to
differentiate, as it does not involve solving the ODE. However, minimizing
``L^\text{prior}`` does not take into account the effect of the prediction
error on the LES solution error. The posterior loss does, but has a longer
computational chain involving solving the LES ODE.

```@autodocs
Modules = [NeuralClosure]
Pages = ["training.jl"]
```

## Neural architectures

We provide neural architectures: A convolutional neural network (CNN), group
convolutional neural networks (G-CNN) and a Fourier neural operator (FNO).

```@autodocs
Modules = [NeuralClosure]
Pages = ["cnn.jl", "groupconv.jl", "fno.jl"]
```

## Data generation

```@autodocs
Modules = [NeuralClosure]
Pages = ["data_generation.jl"]
```
