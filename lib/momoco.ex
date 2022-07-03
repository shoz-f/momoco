defmodule Momoco do
  @moduledoc """
  Documentation for `Momoco`.
  """

  @momoco_path Path.absname("./priv") |> to_charlist()
  @timeout 10000

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
    :python.start({:local, __MODULE__}, python_path: @momoco_path)
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
  
  def to_tensorflow(onnx, path) do
    :python.call(__MODULE__, :momoco, :to_tensorflow, [onnx, path])
  end
  
  def to_tflite(onnx, path) do
    :python.call(__MODULE__, :momoco, :to_tflite, [onnx, path])
  end
  
  def get_tflite(path) do
    :python.call(__MODULE__, :momoco, :get_tflite, [path])
  end
  
  def dummy() do
    :python.call(__MODULE__, :momoco, :dummy, [])
  end
end
