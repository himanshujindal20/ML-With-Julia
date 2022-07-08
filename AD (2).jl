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

# ╔═╡ 1f84feea-bafc-11eb-3320-390dad8b08ab
begin
	using StaticArrays
	using PlutoUI
	import Base: +, *
	using Distributions, Random, Plots, StatsPlots
	plotly()
end

# ╔═╡ 88c7fda2-4633-49c0-9532-008ea8203bae
md"""
# Floating Point gotchas

Floating point arithmetic operations are relatively sized (especially in addition/subtraction). As an example, consider addition of $10^5 + 2.34566266 * 10^{-10}$ as an example.

$$a = 10^5 = 1.0000,0000,0000,0000 * 10^5$$
$$b = 2.34566266*10^{-10} = 0.0000,0000,0000,0234 *10^5$$
and then these two are added. We inherently lose precision if we add large numbers to small numbers. Note that here the floating point operation screws up only when
$\frac{a}{b}$ is too small or too large.

we have,

$ a+b = 1.0000,0000,0000,0234 * 10^5 $
$ (a+b) - a = 0.0000,0000,0000,0234*10^5 $
$ (a+b) - a = 2.34 \;\;\; 81,2547,1436,1433 * 10^{-10}$

"""

# ╔═╡ 7c1c6a94-b3ba-44d0-a2b9-0b9dcbef5747
md"""
# Automatic Differentiation

To start understanding how to compute derivatives on a computer, we start with finite differencing. For finite differencing, recall that the definition of the derivative is:

$$f'(x) = lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x)}{\epsilon}$$

Finite differencing directly follows from this definition by choosing a small $\epsilon$. However, choosing a good  $\epsilon$ is very difficult. 

If  $\epsilon$ is too large than there is error since this definition is asymtopic. 

However, if  $\epsilon$ is too small, you receive roundoff error. To understand why you would get roundoff error, recall that floating point error is relative. Shit gets even more messier if you would have divided it by ϵ after that

![lol](https://www.researchgate.net/profile/Jongrae-Kim-2/publication/267216155/figure/fig1/AS:651888458493955@1532433728729/Finite-Difference-Error-Versus-Step-Size.png)

![](http://degenerateconic.com/wp-content/uploads/2014/11/complex_step1.png)
"""

# ╔═╡ d9f8a465-dd6e-4f07-9754-412937c0f0c2
eps(Float64)

# ╔═╡ 4ab5c9b5-6f36-4a2e-acfa-35b0631509a5
((1 + 10^-10) - 1)

# ╔═╡ 3fcefaaf-f6eb-46df-ae46-9977dba24de0
eps(10^-10)

# ╔═╡ 50b8130b-b0b6-4a8b-b473-080759b9d8ae
md"""
notice the weird one in the corner. How it is acheiving low errors? Well, if taking differences is bad, then don't take differences to get derivatives. Let's see how complex numbers come in here.

$$f(x+ih)=f(x)+ f^′ (x) ih$$

This can yield approximately upto 16 digits of precision. Note that this is different from the differencing case. The derivative shows up in the complex part, hence there is no need to subtract values. Neat. 

$$f'(x) = \frac{Im(f(x+ih))}{h}$$

Let's refine this algorithm further. From next time, we will explicitly set $\epsilon^2 = 0$ wherever we see it (just like we would have $i^2 = -1$), we also define a set of rules to go along with it, such as the following.

In order for this to work out, we need to derive an appropriate algebra for our numbers. To do this, we will look at Taylor series to make our reconstructions. We would get, 

a) $f(a) = f(a) + ϵ f'(a)$

b) $(f+g)=[f(a)+g(a)]+ϵ[f'(a)+g'(a)]$

c) $(f⋅g)=[f(a)⋅g(a)]+ϵ[f(a)⋅g'(a)+g(a)⋅f'(a)]$

all these rules make perfect sense; let's take an example to see that.

let $f(x) = x^2 + 5x$, then $f(x+ϵ) - f(x) = 2xϵ + ϵ^2 + 5ϵ$. ignoring $ϵ^2$ gives us
$f(x+ϵ) = f(x) + ϵ(2x + 5)$, which is exactly of the form $f(x) + f'(x)ϵ$

This method of finding derivatives through pushing epsilons is called forward differentiation.

"""

# ╔═╡ 5104dea4-ad76-41b6-9db2-7aae165f19fb
struct Dual{T}
    val::T   # value
    der::T  # derivative
end

# ╔═╡ bf90b772-79a8-4553-ae92-19e9681ba290
begin
	
	m = Dual(3, 1)
	n = Dual(5, 6)
	m + n
end

# ╔═╡ 8a2d7796-e747-4043-8ef3-eeee822bb40a
f(a) = exp(m)

# ╔═╡ d751cac3-6040-4a64-8281-2ebb61f5280c
f

# ╔═╡ 421ce6b5-b7fd-45c2-a83a-fc669e5377f4
derivative(f, x) = f(Dual(x, one(x))).der # Here, we do one(x) since d\dx(x) = 1 

# ╔═╡ cd72d152-8f72-4cea-8b5c-0d88842dfb3d
derivative(x -> x^2, 5)

# ╔═╡ 8c460054-cb7b-49f6-b3aa-c0a7abcdcd94
md"""
# How do we calculate gradients?

Take $f(x, y) = x^2 + y$,  we first treat $y$ as constant.(Dual number with $0$ in `der` field). then take derivative; then take $x$ as constant, then find derivative. In code.
"""

# ╔═╡ 4895b671-48e1-40e1-a16d-903025957beb
begin
	a, b = 3, 4
	g(x, p) = x^2 + p
	gₓ(x) = g(x, b)
	gₚ(p) = g(a, p)
end

# ╔═╡ ccb31633-1c9d-497e-bbf5-5e8818a33f80
gradient = [derivative(gₓ, a), derivative(gₚ, b)]

# ╔═╡ dd2fe730-f1b7-43fa-9d21-2908d950d0d5
md"""
This substitutes values of $a$ and $b$ as  required. One more way of looking at this would be:
"""

# ╔═╡ fac1dab6-253e-4650-8666-8e3f7fced9e2
@show g(Dual(a, one(a)), Dual(b, zero(b))).der

# ╔═╡ 7191473a-c2aa-486b-aded-a0de932ad0d6
@show g(Dual(a, zero(a)), Dual(b, one(b))).der

# ╔═╡ 925c875a-78fa-458a-a912-47609e9d91f7
md"""
# Can we improve this?

Instead of splitting functions separately, we make ϵ a vector! Each index in ϵ̄ corresponds to a single variable!

What does this mean? for example, in the last function, we do something like

							says, treat y as a variable for which we want the derivative for, but only in the second direction, in the first, treat it as a constant. 	
								|
	f(Dual(x, (1, 0)), Dual(y, (0, 1))) 
	            |
			Basically says, treat x as a variable for which we want a derivative in the first direction, in the second direction, treat it as a constant.

"""

# ╔═╡ 8863529e-4237-4c46-bfff-5eb6cae8d37c

struct MultiDual{N,T} # N-> Size of the vector, T -> DataType of the vector
    val::T
    derivs::SVector{N,T}
end

# ╔═╡ 8f63c6dc-2d63-4ec2-a3c6-116a806352bb
begin
	Base.:+(f::Dual, g::Dual) = Dual(f.val + g.val, f.der + g.der)
	Base.:+(f::Dual, α::Number) = Dual(f.val + α, f.der)
	Base.:+(α::Number, f::Dual) = f + α

	Base.:-(f::Dual, g::Dual) = Dual(f.val - g.val, f.der - g.der)

	# Product Rule
	Base.:*(f::Dual, g::Dual) = Dual(f.val*g.val, f.der*g.val + f.val*g.der)
	Base.:*(α::Number, f::Dual) = Dual(f.val * α, f.der * α)
	Base.:*(f::Dual, α::Number) = α * f

	# Quotient Rule
	Base.:/(f::Dual, g::Dual) = Dual(f.val/g.val, (f.der*g.val - f.val*g.der)/(g.val^2))
	Base.:/(α::Number, f::Dual) = Dual(α/f.val, -α*f.der/f.val^2)
	Base.:/(f::Dual, α::Number) = f * inv(α) # Dual(f.val/α, f.der * (1/α))

	Base.:^(f::Dual, n::Integer) = Base.power_by_squaring(f, n)
	Base.:exp(f::Dual) = Dual(exp(f.val), exp(f.val) * f.der)
	
######################################################################################
	
	function Base.:+(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N,T}
    	return MultiDual{N,T}(f.val + g.val, f.derivs + g.derivs)
	end

	function Base.:*(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N,T}
    	return MultiDual{N,T}(f.val * g.val, f.val .* g.derivs + g.val .* f.derivs)
	end
	
	Base.:^(f::MultiDual{N, T}, n::Integer) where {N,T} = Base.power_by_squaring(f, n)
	

	function Base.:*(f::MultiDual{N,T}, α::Number) where {N,T}
    	return MultiDual{N,T}(f.val * α, f.derivs .* α)
	end

	Base.:*(α::Number, f::MultiDual{N, T}) where {N, T} = f*α	

end

# ╔═╡ 626cbcd1-d142-4300-9569-337050471087
md"""
Let's take a simple example to see what is going on. let $f(x, y) = x^2 + y^2$. The steps of calculation are as follows.

```julia
x = MultiDual(3, (1, 0))
y = MultiDual(4, (0, 1))
x^2 = x*x # This calls * method defined above.
y^2 = y*y # This calls * method defined above.
x^2 + y ^2 # calls + method above
```

- The above function will return `(9, (6, 0))` for x, and `(16, (0, 8))`.
- Adding them gives us `(25, (6, 8))`, which is what we need!
"""

# ╔═╡ a9f815da-6b18-427d-b97b-313af0ed12ce
h(x, y) = (2x + y)^2

# ╔═╡ f6b170ed-db73-4041-85df-0ef9f13975d1
begin
	u = MultiDual(3, SVector(1, 0))
	v = MultiDual(4, SVector(0, 1))
end

# ╔═╡ 3eaf826e-59d0-4882-9355-a81e160200db
h(u, v)

# ╔═╡ 3496ba85-8010-4451-a328-20c5899520c4
md"""
# Reverse Differentiation

Just like forward diff was calculated by pushing ϵ's, In reverse differentiation we use chain rule to make our lives easier. We will not be implementing it, because it is slightly more involved. 

##### Computational Graph:

Best if I show it through an example:

Lets say we have 2 variables $a$ and $b$, $f(a,b) = e = (a+b)(b+1)$. The computational graph looks like:

$(LocalResource("./Backpropbasic.png"))

After adding derivatives, we have

$(LocalResource("./diff_backprop.png"))

"""

# ╔═╡ afce7888-cdde-4004-ba56-1f3b95d88e7c
md"""
Here is one more example:
$(LocalResource("./Backprop1.png"))
"""

# ╔═╡ c8b826d0-4d26-4c52-8d4d-683143cbb42b
md"""
Here is one more!, we close AD with this one.
$(LocalResource("./Backprop2.png"))
"""

# ╔═╡ 8728406a-acfb-4507-9f6d-24d9a09d6bae
md"""
## Probability and MLE: Not too formal stuff.

### Prelims

- Dependant events: $p(y|x) = \frac{p(y \cap x)}{p(x)}$
- Independent events $p(y \cap x) = p(x)p(y)$
- Bayes Rule $p(y|x) = \frac{p(x|y)p(y)}{p(x)}$

### Probability Distribution:

**random variable**: Basically we associate each event with a number.

- For example: in a coin toss, we can map the outcome of heads to 1 and tails to 0, and name this random variable as $X$. Then we would write something like:

- P(X=1) = 1/2
- P(X=0) = 1/2

**Distribution of a random variable**

- This is a function that maps each random variable to it's probability. For example, the probability distribution of $X$ in the prev example is given by:

#### Discrete Random Variables:

- In this type of rv, X is constrained to take discrete values only, albeit these might be infinite. Let's take a look at a few examples.

###### Bernoulli Distribution:

The distribution that is formed when you toss a coin whose probability for showing heads is given by $p$. Let X be a rv as defined in the previous case. **Then X is said to follow Bernoulli distribution parametrized by $p$ if it's probability distribution is as follows. In short you write, $X \sim Bern(p)$** 

- P(X = 1) = $p$
- P(X= 0) = $1-p$
or,
- P(X= $x$) = $p^x (1-p)^{1-x}$

##### Binomial Distribution:

Consider the example of tossing a coin $n$ times  independently whose P(heads) = $p$. Let X be a rv denoting the numbers of heads in $n$ tosses. **Then X is said to follow Binomial distribution parametrized by $p$ and $n$ if it's probability distribution is as follows. In short $ X \sim Bin(n, p)$**

- P(X = x) = $\binom{n}{x}p^x (1-p)^{n-x}$

# Important
**Sum of probability over all $x$ is $1$.** 

The probability distribution is also called the probability mass function.

## Relation with point masses? Jamboard.

Expectation of a random variable is just the center of mass of the object.

Variance is just the moment of inertia about the center of mass.

## What happens when there is density instead of point masses?
"""

# ╔═╡ a42403cc-322b-498d-b02a-d2036c7f75dd
md"""
## Continous random variables.

Here the random variable can take continous values in a given range. Let's take a look at a few examples. The distribution of such a variable is not surprisingly, called probability density function.

Here too, we have the same condition:

##### Uniform Distribution:

This is when the density of the rod is uniform, when the rod exists from $a$ to $b$.
Formally, $X \sim U(a, b)$

expectation: $\frac{a + b}{2}$

variance: $\frac{(b-a)^2}{12}$ 
##### Normal Distribution:

Infinitely long rod, with density as follows. Formally we say,  $ X \sim N(\mu, \sigma^2)$

density at any point $x$ is given by

$$f_X(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{(x - \mu)}{2\sigma^2}}$$

expectation: $\mu$

variance: $\sigma^2$

Let's move onto computer simulations
"""

# ╔═╡ 5ecc374c-c310-4d1e-9399-7d330e0ab50a
Random.seed!(1234)

# ╔═╡ 9f7b9064-eaa4-4c11-8df4-c6ad2020cf02
md"""
p : $(@bind p Slider(0:0.01:1, default=0.5))

nn: $(@bind nn Slider(0:100, default=50))

pts: $(@bind logpt Slider(1:5, default=2))
"""

# ╔═╡ 20cab814-d7c5-456f-9ed9-a69a21a9a8fc
pts = 10^logpt

# ╔═╡ 3536a911-b288-4dc4-8bda-0ca7c0edabe9
d = Bernoulli(p)

# ╔═╡ 826a9638-3ded-4a32-b8ea-8d9daff9cc00
rand(d, 5)

# ╔═╡ fae74109-603e-42ce-9f1d-d18ec94b9a61
histogram(rand(Binomial(nn, p), pts),bins=nn)

# ╔═╡ bd3a1223-c356-42ff-aa72-7d1c3dde15ad
md"""
μ: $(@bind μ Slider(-5:0.2:5, default=0))
σ²: $(@bind σ² Slider(1:0.5:10, default=5))

a: $(@bind α Slider(1:0.5:10, default=5))
b: $(@bind β Slider(15:0.5:25, default=20))
"""

# ╔═╡ 674d4200-dce1-4d09-98c4-1a9cb3be9c3f
plot(Normal(μ,σ²), xlims=(-20, 20))

# ╔═╡ df1248ec-b893-4da3-8efb-a45cc4fd1d0d
plot(Uniform(α, β), ylims=(0,0.25))

# ╔═╡ e80fef5d-c8fc-437d-8a2e-45d0d96a58e1
md"""
## MLE: Maximum Likelihood Estimation:

We have seen how to get random samples from a probability distribution $p(X)$, Let us now take a look at the reverse problem!

We are given $X_1, X_2, \ldots X_n$ samples from $N(\theta, \sigma^2)$. Can we find the best possible fit for $\theta$ and $\sigma^2$??


Let's take a visual example. Here we'll be considering θ as the variable(σ² is given)
"""

# ╔═╡ e0df2b0a-b9b0-47ed-8ef4-a4cb25b35c65
X = rand(Normal(3, 2), 1)

# ╔═╡ 9a672f4f-292e-4cf6-bbf1-8437b70a0228
md"""
mu: $(@bind mu Slider(-3:0.5:4) )
sigmasq: $(@bind sigmasq Slider(1:0.5:10))
"""

# ╔═╡ 1ed3b357-029c-48a0-b7d8-c94bf51c11db
md"""
mu: $mu sigmasq: $sigmasq
"""

# ╔═╡ 8d5776a3-dccd-4163-99c3-70cbf24a42b3
begin
	scatter(X, X-X, xlims=(-4,10))
	plot!(Normal(mu, sigmasq))
end

# ╔═╡ a2fd29d9-f598-4874-9619-ac295488cc27
md"""
The Maximum Likelihood Estimator is one way to solve the problem.

Likelihood: Value of $f_X(x)$ at any point.

IE: Find theta and sigma such that likelihood is maximised. You saw an example when we only have one sample. When we have multiple samples,we just multiply them togather.

$$L(X|\theta, \sigma) = \Pi_{i=1}^n f_X(x_i)$$, in this case, we get.


L(X|θ, σ) = K₁ exp(K₂ Σᵢ (xᵢ - θ)²)

we need to find θ such that above is maximised.

We take log on both sides, the maximizing function becomes:

l(X|θ, σ) = ln(K₁) + K₂ Σᵢ (xᵢ - θ)²

Easy to differentiate and see that $θ = \sum_i \frac{x_i}{N}$


## Exercise: derive what happens when both sigma and theta are variable.
"""

# ╔═╡ 2d5b6bb1-03cc-4a24-b943-774edbf7269a
md"""
# What happens when there is no solution to the derivative?
"""

# ╔═╡ Cell order:
# ╠═1f84feea-bafc-11eb-3320-390dad8b08ab
# ╟─88c7fda2-4633-49c0-9532-008ea8203bae
# ╟─7c1c6a94-b3ba-44d0-a2b9-0b9dcbef5747
# ╠═d9f8a465-dd6e-4f07-9754-412937c0f0c2
# ╠═4ab5c9b5-6f36-4a2e-acfa-35b0631509a5
# ╠═3fcefaaf-f6eb-46df-ae46-9977dba24de0
# ╟─50b8130b-b0b6-4a8b-b473-080759b9d8ae
# ╠═5104dea4-ad76-41b6-9db2-7aae165f19fb
# ╠═8f63c6dc-2d63-4ec2-a3c6-116a806352bb
# ╠═bf90b772-79a8-4553-ae92-19e9681ba290
# ╠═8a2d7796-e747-4043-8ef3-eeee822bb40a
# ╠═d751cac3-6040-4a64-8281-2ebb61f5280c
# ╠═421ce6b5-b7fd-45c2-a83a-fc669e5377f4
# ╠═cd72d152-8f72-4cea-8b5c-0d88842dfb3d
# ╟─8c460054-cb7b-49f6-b3aa-c0a7abcdcd94
# ╠═4895b671-48e1-40e1-a16d-903025957beb
# ╠═ccb31633-1c9d-497e-bbf5-5e8818a33f80
# ╟─dd2fe730-f1b7-43fa-9d21-2908d950d0d5
# ╠═fac1dab6-253e-4650-8666-8e3f7fced9e2
# ╠═7191473a-c2aa-486b-aded-a0de932ad0d6
# ╟─925c875a-78fa-458a-a912-47609e9d91f7
# ╠═8863529e-4237-4c46-bfff-5eb6cae8d37c
# ╟─626cbcd1-d142-4300-9569-337050471087
# ╠═a9f815da-6b18-427d-b97b-313af0ed12ce
# ╠═f6b170ed-db73-4041-85df-0ef9f13975d1
# ╠═3eaf826e-59d0-4882-9355-a81e160200db
# ╟─3496ba85-8010-4451-a328-20c5899520c4
# ╟─afce7888-cdde-4004-ba56-1f3b95d88e7c
# ╟─c8b826d0-4d26-4c52-8d4d-683143cbb42b
# ╟─8728406a-acfb-4507-9f6d-24d9a09d6bae
# ╠═a42403cc-322b-498d-b02a-d2036c7f75dd
# ╠═5ecc374c-c310-4d1e-9399-7d330e0ab50a
# ╠═9f7b9064-eaa4-4c11-8df4-c6ad2020cf02
# ╟─20cab814-d7c5-456f-9ed9-a69a21a9a8fc
# ╠═3536a911-b288-4dc4-8bda-0ca7c0edabe9
# ╠═826a9638-3ded-4a32-b8ea-8d9daff9cc00
# ╠═fae74109-603e-42ce-9f1d-d18ec94b9a61
# ╠═bd3a1223-c356-42ff-aa72-7d1c3dde15ad
# ╠═674d4200-dce1-4d09-98c4-1a9cb3be9c3f
# ╠═df1248ec-b893-4da3-8efb-a45cc4fd1d0d
# ╠═e80fef5d-c8fc-437d-8a2e-45d0d96a58e1
# ╠═e0df2b0a-b9b0-47ed-8ef4-a4cb25b35c65
# ╟─9a672f4f-292e-4cf6-bbf1-8437b70a0228
# ╟─1ed3b357-029c-48a0-b7d8-c94bf51c11db
# ╠═8d5776a3-dccd-4163-99c3-70cbf24a42b3
# ╟─a2fd29d9-f598-4874-9619-ac295488cc27
# ╟─2d5b6bb1-03cc-4a24-b943-774edbf7269a
