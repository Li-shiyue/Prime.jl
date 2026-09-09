# Prime.jl

<p align="center">
  <img src="sc.png" width="850">
</p>

**Prime.jl** is a Julia package that implements a **cell clustering method based on a biophysical gene-expression model**, formulated under the **Probability Generating Function (PGF)** framework.

The package provides a concise public interface with two main entry-point functions:

- **`prime_cluster`**: performs **cell clustering** under the PGF-based modeling framework.
- **`prime_infer_to_csv`**: performs **Bayesian inference** of **gene- and cluster-specific model parameters**, and **exports the inferred parameters to CSV** for downstream analysis and quality control.

A complete, runnable usage example is provided here:  
https://github.com/Li-shiyue/Prime.jl/blob/main/examples/quickstart.jl

## 1) Install Julia

Prime.jl is developed and tested with **Julia 1.11.x**.

### 1. Download Julia (official installer)
Download the Julia installer for your operating system from the official website:  
https://julialang.org/downloads/

### 2. Install Julia (platform guide)
Follow the official platform-specific installation instructions:  
https://julialang.org/downloads/oldreleases/

### 3. Verify your Julia installation
After installation, confirm that Julia is available in your terminal:

```bash
julia --version
```

## 2) Install Prime.jl 
Prime.jl can be installed directly from this GitHub repository (no General registry registration required).
Start Julia and run:

```julia
using Pkg
Pkg.develop(PackageSpec(url="https://github.com/Li-shiyue/Prime.jl.git"))
using Prime
