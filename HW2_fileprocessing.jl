# New chunk: function that takes the filename as an argument.
# Currently, returns matrixes of v and m values, instance names.
# Eventually, include GA calls within function

function GApartA(fileName)
    regExp = r"knapPI_(?<t>\d+)_(?<n>\d+)_(?<R>\d+)"
    m = match(regExp,fileName)
    n = parse(Int, m[2])
    
    open(fileName) do file
        fileText = readlines(file)
        lineCount = 1
        instanceName = []
        vAll = zeros(Int,100,n)
        wAll = zeros(Int,100,n)
        for i in 1:100
            push!(instanceName,fileText[lineCount])
            lineCount += 4 #skips unneeded lines
            for j in 1:n
                vw = match(r"\d+,(?<v>\d+),(?<w>\d+)",fileText[lineCount + j])
                vAll[i,j] = parse(Int, vw[1])
                wAll[i,j] = parse(Int, vw[2])
            end
            lineCount += n #skip to end of instance
            lineCount += 3 #skips the filler lines between instances
        end
        
        return instanceName, vAll, wAll
    end

end