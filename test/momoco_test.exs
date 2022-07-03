defmodule MomocoTest do
  use ExUnit.Case
  doctest Momoco

  test "greets the world" do
    assert Momoco.hello() == :world
  end
end
