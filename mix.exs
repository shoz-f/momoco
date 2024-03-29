defmodule Momoco.MixProject do
  use Mix.Project

  def project do
    [
      app: :momoco,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Momoco.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:erlport, "~> 0.10.1"},
      {:onnxs, "~> 0.2.0"},
      {:eflatbuffers, "~> 0.1.0"},
      {:axon_onnx, github: "elixir-nx/axon_onnx"},
    ]
  end
end
