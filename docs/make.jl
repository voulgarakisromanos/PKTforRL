using Documenter
using PKTforRL

makedocs(; sitename="PKTforRL", format=Documenter.HTML(), modules=[PKTforRL])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
