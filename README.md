**For SSH Tunneling** : ```ssh -L 8080:localhost:8501 -J hardik.mittal@ada gnode053```

<br>

**For opening gnode in vs code** : ```ssh -J hardik.mittal@ada hardik.mittal@gnode053```

<br>

**If streamlit doesn't have a path** : ```export PATH="$HOME/.local/bin:$PATH"```


**Set the torch cache folder** : 
* ```export TORCH_HOME=<path to the cache folder>```
* ```export TRANSFORMERS_CACHE=<path to the cache folder>```



