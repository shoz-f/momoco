defmodule Momoco do
  @moduledoc """
  Documentation for `Momoco`.
  """

  @momoco_path Path.absname("./priv") |> to_charlist()

  def child_spec(opts) do
    %{
      id: __MODULE__,
      start: {__MODULE__, :start_link, [opts]},
      type: :worker,
      restart: :permanent,
      shutdown: 500
    }
  end

  def start_link(_args \\ []) do
    :python.start({:local, __MODULE__}, python: 'python3', python_path: @momoco_path)
    |> IO.inspect()
  end

  def stop() do
    :python.stop(__MODULE__)
  end

  def python(mod, func, args) do
    :python.call(__MODULE__, mod, func, args)
  end
  
  def load_onnx(path) do
    :python.call(__MODULE__, :momoco, :load_onnx, [path])
  end
  
  def save_onnx(onnx, path) do
    :python.call(__MODULE__, :momoco, :save_onnx, [onnx, path])
  end

  def to_tensorflow(onnx, path) do
    :python.call(__MODULE__, :momoco, :to_tensorflow, [onnx, path])
    path
  end
  
  def to_tflite(onnx, path) do
    to_tensorflow(onnx, path)
    |> to_tflite(path)
  end
  
  def to_tflite(path) do
    :python.call(__MODULE__, :momoco, :to_tflite, [path])
    path
  end

  def to_torch(onnx, path) do
    :python.call(__MODULE__, :momoco, :to_torch, [onnx, path])
  end

  def from_saved_model(path) do
    :python.call(__MODULE__, :momoco, :from_saved_model, [path])
  end
  
  def from_tflite(path) do
    :python.call(__MODULE__, :momoco, :from_tflite, [path])
  end
end
