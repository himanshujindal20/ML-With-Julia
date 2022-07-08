### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ d55af3ce-88e7-11eb-1fff-b1598f592982
begin
	using Zygote
	using Plots
	using PlutoUI
	using LinearAlgebra
	plotly()
end

# ╔═╡ d59b6eb7-5df1-4aea-8036-dc44cba5553b
md"""
# Gradient Descent:

- Gradient gives the direction of steepest ascent.
- -1 * Gradient Gives the direction of steepest descent
- If we take a small step in that direction, the value of our function ought to decrease(Atleast we hope so)

So formally speaking,  it is put togather like this.

- we want to minimize $f(x)$
- We cannot directly find $x$ by solving ∇f(x) = $0$
- We have an initial guess $x_1$
- We find gradient $\nabla f(x_1)$
- we update $x_1$ as follows:

$$x_2 = x_1 - ϵ\nabla f(x_1)$$

- We keep doing this for n iterations, hopefully we arrive at minimum.

"""

# ╔═╡ 73ffc44c-88e7-11eb-3a22-d114c30e5f8d
begin
    import DarkMode
    DarkMode.enable()
end

# ╔═╡ 34a2efb6-88e9-11eb-3473-f5a944847260
begin
	mutable struct SGD
    	η::Float64
	end
	SGD() = SGD(η=0.1)
end

# ╔═╡ 5b4f0c6c-88e9-11eb-39b8-d53fce4ae55c
function descend(opt::SGD, xₜ::Vector{T}, ∇xₜ::Vector{T}) where T

    ∇xₜ .*= opt.η
    xₜ .-= ∇xₜ

end

# ╔═╡ 7e29da96-88e9-11eb-2bf0-097f420d20c6
function optimize!(opt::SGD, x::Vector{T}, f::Function, n::Int) where T

	Y = Vector{T}[]
	for i in 1:n
		push!(Y, x)
		∇x  = gradient(z -> f(z) , x)[1]
		x = x - opt.η .* ∇x
	end
	return Y
end

# ╔═╡ 70dc4750-88eb-11eb-2a2a-dd2315f4398d
md"""
x1 : $(@bind x₁ PlutoUI.Slider(-2:0.05:2, show_value = true, default=1.0))
x2 : $(@bind x₂ PlutoUI.Slider(-2:0.05:2, show_value = true,default=1.0))
"""

# ╔═╡ 9527e48a-897b-11eb-1014-b3f559433ade
md"""
η: $(@bind η Slider(0.01:0.01:0.3, show_value =true, default=0.02))
iterations: $(@bind its Slider(1:50 , show_value =true, default=5))
"""

# ╔═╡ 6fcf5d72-894d-11eb-28b0-a71a25aea9e8
x = Float64[x₁, x₂]

# ╔═╡ 4761b40a-88eb-11eb-1732-d78df92902fd
opt = SGD(η)

# ╔═╡ 9745a65c-732b-445b-ac52-71a1a04b8bb7
begin
	f(x::Vector{T}) where T = x[1]^4 + x[2]^3
	f(x, y) = f([x, y])
end

# ╔═╡ 6a3c37d0-88ef-11eb-37e4-4daea578fc5d
Y = optimize!(opt, x, f, its)

# ╔═╡ fb399200-897b-11eb-0f15-d95305b4bbe8
z = range(-2, stop = 2, length=100)

# ╔═╡ 0845b032-897e-11eb-3bfd-35828ff0cb8c
begin
	y₁ = map(y -> y[1], Y)
	y₂ = map(y -> y[2], Y)
end

# ╔═╡ e69eafce-8953-11eb-3f3c-312cc271cec2
begin
	plot(z, z, f, st=:surface, c=cgrad(:greens))
	plot!(y₁, y₂, f.(Y), st=[:line, :scatter], markersize = 2,  linecolor=:blue, markercolor=:red, markershape=:cross, markeralpha=0.8, linewidth=2)
end

# ╔═╡ Cell order:
# ╟─d59b6eb7-5df1-4aea-8036-dc44cba5553b
# ╠═73ffc44c-88e7-11eb-3a22-d114c30e5f8d
# ╠═d55af3ce-88e7-11eb-1fff-b1598f592982
# ╠═34a2efb6-88e9-11eb-3473-f5a944847260
# ╠═5b4f0c6c-88e9-11eb-39b8-d53fce4ae55c
# ╠═7e29da96-88e9-11eb-2bf0-097f420d20c6
# ╠═70dc4750-88eb-11eb-2a2a-dd2315f4398d
# ╟─9527e48a-897b-11eb-1014-b3f559433ade
# ╠═6fcf5d72-894d-11eb-28b0-a71a25aea9e8
# ╠═4761b40a-88eb-11eb-1732-d78df92902fd
# ╠═9745a65c-732b-445b-ac52-71a1a04b8bb7
# ╟─6a3c37d0-88ef-11eb-37e4-4daea578fc5d
# ╠═fb399200-897b-11eb-0f15-d95305b4bbe8
# ╟─0845b032-897e-11eb-3bfd-35828ff0cb8c
# ╠═e69eafce-8953-11eb-3f3c-312cc271cec2
