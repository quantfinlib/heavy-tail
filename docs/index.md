# [heavy<sub>tail</sub>]


[🌐 **GitHub**](https://github.com/quantfinlib/heavy-tail)
&nbsp;&nbsp;&nbsp; [🔗 **API**](heavytail)
&nbsp;&nbsp;&nbsp; [📖 **Docs**](https://quantfinlib.github.io/heavy-tail/)

## Documentation

The documentation is available at [githubpages](https://quantfinlib.github.io/heavy-tail/).
The [🔗 API documentation](heavytail) is generated using [pdoc3](https://pdoc3.github.io/pdoc/).

To manually generate the documentation, first, install the heavytail package with the doc dependencies using `uv`:
 
```bash
$ uv pip install -e .[docs]
```

Then

```bash
$ uv run pdoc --html  -c latex_math=True --output-dir docs --force heavytail
```