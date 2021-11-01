# To run this, you need to install Julia 1.6, and install the packages listed on the next line.
# This can be done by pressing "]" and typing add *package name*, then backspace to return to
# the julia REPL.
using Evolutionary, CSV, DataFrames, Distributions, HypothesisTests, Printf, Plots, StatsPlots

# Function to return the fitness and function calls from each of the 30 replicates for each of the 81 combinations.
# Necessary in each of part a's functions, so I've defined it here.
function stats(l)
    temp=map(y->mapreduce(x->hcat(-minimum(x),Evolutionary.iterations(x),Evolutionary.f_calls(x)),vcat,l[y]),1:30)
    fit=map(y->map(x->temp[x][y],1:30),1:81)
    calls=map(y->map(x->temp[x][y],1:30),163:243)
    return fit, calls
end

#Create bitvector of zeros to represent an empty solution. These are randomized in the population initialization.
# This was the best performing initialization method, but the alternatives are still present in
# the exploratory_heatmaps function.
function x0(n)
    return flip(BitVector(zeros(n)))
end

# Define functions for part A.
function parta_exploratory_heatmaps(file,fname="")
    f_in(x)=SubString(x,8,length(x)-4)
    inst=f_in(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false)

    #Also read in constraint value
    con=open(file) do file
        fileText= readlines(file)
        con=fileText[3][3:end]
        return parse(Int,con)
    end

    #Create variables, knapsack options (n), weights (w), and values (v)
    n=size(data)[1]
    w=data[:,3]
    v=data[:,2]

    #Create objective function, which defaults to 0 if it violates the constraint,
    #and the sum is set to negative as this EA uses minimization
    # This constraint handling technique was the best for
    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : (sum(w.*x)-con))

    # Alternative constraint handling attempt.
    #f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : 0)

    #Alternative initialization conditions
    #x0=flip(BitVector(zeros(n)))
    #x0=BitVector(rand(Bool,n)) function x0(n)

    #Defined uniform bit flipping function.
    #Flips a bit with probability 1/length(genome)
    function uflip(recombinant::T) where {T <: BitVector}
        s = length(recombinant)
        p = 1/s
        check = rand(s).<p
        for i in 1:length(check)
            if check[i]==1
            recombinant[i]=!recombinant[i]
            end
        end
        return recombinant
    end

    #Define different combinations of genetic algorithms with different types of mutation, and types of crossover
    #uniform and singlepoint.
    # Population size is pinned to 4*genome length, and mutation rate minimum/maximum are related to this value according to past work at purdue CITE.
    # Crossover rates "average" tends to be 0.5, so we sampled from 0.1..0.9
    popsize=4*n
    points=mapreduce(y->map(x->(x,y),LinRange(1/(popsize*n),1/popsize,9)),vcat,0.1:0.1:0.9)
    gafu=map(x->GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    gafs=map(x->GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    gauu=map(x->GA(populationSize=popsize,selection=roulette,mutation=uflip,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    gaus=map(x->GA(populationSize=popsize,selection=roulette,mutation=uflip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    #Run the optimization 30 times for each of the 81 combinations, collect distributions on fitness and calls, suppress output.
    rfu = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafu) for _ in 1:30];
    fu_fitness, fu_calls = stats(rfu);

    rfs = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafs) for _ in 1:30];
    fs_fitness, fs_calls = stats(rfs);

    ruu = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gauu) for _ in 1:30];
    uu_fitness, uu_calls = stats(ruu);

    rus = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gaus) for _ in 1:30];
    us_fitness, us_calls = stats(rus);

    #Define X and Y values for heatmap plots, with X corresponding to Mutation Rate, and Y to Crossover Rate
    xs = [@sprintf("%.1E",i) for i = LinRange(1/(popsize*n),1/popsize,9)]
    ys = [string("C", i) for i = 0.1:0.1:0.9]

    # This coerces data into the appropriate matrix form for creating heatmaps.
    function hm(data,f)
        return reshape(f.(data),9,:)'
    end

    # This and the following function allows easy plotting of the heatmaps, and manipulating the title values
    function f_plotter(data,fname,instance)
        plot(data[1],data[2],data[3],data[4], layout=4, tickfont=(3), titlefont=(8), plot_title=string(fname," Scores for", " ", instance))
    end

    function c_plotter(data,fname,instance)
        plot(data[1],data[2],data[3],data[4], layout=4, tickfont=(3), titlefont=(8), plot_title=string(fname," Calls for", " ", instance))
    end

    # Vectorized fitness and function calls for the previous plot functions
    fits=[fu_fitness,fs_fitness,uu_fitness,us_fitness]
    calls=[fu_calls,fs_calls,uu_calls,us_calls]

    #Create heatmaps for example plots, specifically to determine the best parameters for each setup across median, best, and worst performances.
    #Also investigate the most frequently occurring values. Applied to both the fitness and number of function calls.
    med_fit=map(x->hm(x,median),fits)
    med_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(median.(x)))),med_fit)
    mod_fit=map(x->hm(x,mode),fits)
    mod_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(mode.(x)))),mod_fit)
    max_fit=map(x->hm(x,maximum),fits)
    max_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(max.(x)))),max_fit)
    min_fit=map(x->hm(x,minimum),fits)
    min_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(minimum.(x)))),min_fit)
    med_calls=map(x->hm(x,median),calls)
    med_calls_hm=map(x->heatmap(xs,ys,x),med_calls)
    mod_calls=map(x->hm(x,mode),calls)
    mod_calls_hm=map(x->heatmap(xs,ys,x),mod_calls)
    max_calls=map(x->hm(x,maximum),calls)
    max_calls_hm=map(x->heatmap(xs,ys,x),max_calls)
    min_calls=map(x->hm(x,minimum),calls)
    min_calls_hm=map(x->heatmap(xs,ys,x),min_calls)

    #Create output directory for figures
    if(fname!="")
        foldername=fname
    else
        foldername="figures"
    end

    # Create directory if it doesn't exist
    if !isdir(foldername)
        mkdir(foldername)
    end

    # Plot and save fitness heatmaps
    m1=f_plotter(med_fit_hm,"Median", string("Instance ", inst))
    m2=f_plotter(max_fit_hm,"Maximum", string("Instance ", inst))
    m3=f_plotter(mod_fit_hm,"Mode", string("Instance ", inst))
    m4=f_plotter(min_fit_hm,"Minimum", string("Instance ", inst))
    savefig(m1,string("./",foldername,"/Median_",inst))
    savefig(m2,string("./",foldername,"/Maximum_",inst))
    savefig(m3,string("./",foldername,"/Mode_",inst))
    savefig(m4,string("./",foldername,"/Minimum_",inst))

    # Plot and save calls heatmaps.
    m1=c_plotter(med_calls_hm,"Median", string("Instance ", inst))
    m2=c_plotter(max_calls_hm,"Maximum", string("Instance ", inst))
    m3=c_plotter(mod_calls_hm,"Mode", string("Instance ", inst))
    m4=c_plotter(min_calls_hm,"Minimum", string("Instance ", inst))
    savefig(m1,string("./",foldername,"/Median_calls_",inst))
    savefig(m2,string("./",foldername,"/Maximum_calls_",inst))
    savefig(m3,string("./",foldername,"/Mode_calls_",inst))
    savefig(m4,string("./",foldername,"/Minimum_calls_",inst))
end

function parta_boxplots(file,fname="")
    f_in(x)=SubString(x,8,length(x)-4)
    inst=f_in(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false)

    #Also read in constraint value
    con=open(file) do file
        fileText= readlines(file)
        con=fileText[3][3:end]
        return parse(Int,con)
    end

    #Create variables, knapsack options (n), weights (w), and values (v)
    n=size(data)[1]
    w=data[:,3]
    v=data[:,2]

    #Best objective function and constraint method, penalty = how much we've violated the constraint.
    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : (sum(w.*x)-con))

    #Define different combinations of genetic algorithms with different types of mutation, and types of crossover
    #uniform and singlepoint.
    # Population size is pinned to 4*genome length, and mutation rate minimum/maximum are related to this value according to past work at purdue CITE.
    # Crossover rates "average" tends to be 0.5, so we sampled from 0.1..0.9
    popsize=4*n
    points=mapreduce(y->map(x->(x,y),LinRange(1/(popsize*n),1/popsize,9)),vcat,0.1:0.1:0.9)
    gafs=map(x->GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    #Run the optimization 30 times for each of the 81 combinations, collect distributions on fitness and calls, suppress output.
    rfs = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafs) for _ in 1:30];
    fs_fitness, fs_calls = stats(rfs);

    #Define X and Y values for heatmap plots, with X corresponding to Mutation Rate, and Y to Crossover Rate
    xs = [@sprintf("%.1E",i) for i = LinRange(1/(popsize*n),1/popsize,9)]
    ys = [string("C", i) for i = 0.1:0.1:0.9]

    # This and the following function allows easy plotting of the heatmaps, and manipulating the title values
    ffmrm=reduce(hcat,map(x->reduce(vcat,fs_fitness[x:9:x+72]),1:9))
    ffcrm=reduce(hcat,map(x->reduce(vcat,fs_fitness[x:x+8]),1:9:73))
    fcmrm=reduce(hcat,map(x->reduce(vcat,fs_calls[x:9:x+72]),1:9))
    fccrm=reduce(hcat,map(x->reduce(vcat,fs_calls[x:x+8]),1:9:73))

    #Create output directory name figures
    if fname!=""
        foldername=fname
    else
        foldername="figures"
    end

    # Create directory if it doesn't exist
    if !isdir(foldername)
        mkdir(foldername)
    end

    #Save figures to directory.
    savefig(boxplot(permutedims(xs),ffmrm,leg=false, title="Box Plot of Fitness vs Mutation Rate"),string("./",foldername,"/Box_Fit_MR",inst))
    savefig(boxplot(permutedims(ys),ffcrm,leg=false, title="Box Plot of Fitness vs Crossover Rate"),string("./",foldername,"/Box_Fit_CR",inst))
    savefig(boxplot(permutedims(xs),fcmrm,leg=false, title="Box Plot of Function Calls vs Mutation Rate"),string("./",foldername,"/Box_Calls_MR",inst))
    savefig(boxplot(permutedims(ys),fccrm,leg=false, title="Box Plot of Function Calls vs Crossover Rate"),string("./",foldername,"/Box_Calls_CR",inst))

end

parta_boxplots.(files,"test")

function parta_KW_tests(file)
    file=files[1];
    f_in(x)=SubString(x,8,length(x)-4)
    inst=f_in(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false)

    #Also read in constraint value
    con=open(file) do file
        fileText= readlines(file)
        con=fileText[3][3:end]
        return parse(Int,con)
    end

    #Create variables, knapsack options (n), weights (w), and values (v)
    n=size(data)[1]
    w=data[:,3]
    v=data[:,2]

    #Best constraint handling option: penalty based on how much we violate the constraint.
    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : (sum(w.*x)-con))

    #Define different combinations of genetic algorithms with different types of mutation, and types of crossover
    #uniform and singlepoint.
    # Population size is pinned to 4*genome length, and mutation rate minimum/maximum are related to this value according to past work at purdue CITE.
    # Crossover rates "average" tends to be 0.5, so we sampled from 0.1..0.9
    popsize=4*n
    points=mapreduce(y->map(x->(x,y),LinRange(1/(popsize*n),1/popsize,9)),vcat,0.1:0.1:0.9)
    gafs=map(x->GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    #Run the optimization 30 times for each of the 81 combinations, collect distributions on fitness and calls, suppress output.
    rfs = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafs) for _ in 1:30];
    fs_fitness, fs_calls = stats(rfs);

    # This block lets is for hypothesis tests to see which distributions are changing with which parameters.
    ffmrm=map(x->reduce(vcat,fs_fitness[x:9:x+72]),1:9)
    ffcrm=map(x->reduce(vcat,fs_fitness[x:x+8]),1:9:73)
    fcmrm=map(x->reduce(vcat,fs_calls[x:9:x+72]),1:9)
    fccrm=map(x->reduce(vcat,fs_calls[x:x+8]),1:9:73)
    map(x->pvalue(KruskalWallisTest(x ...)),[ffmrm,ffcrm,fcmrm,fccrm])
end

#Set Filename and Read CSV into DataFrame
files=["knapPI_11_100_1000.csv","knapPI_13_200_1000.csv","knapPI_13_50_1000.csv","knapPI_14_50_1000.csv","knapPI_15_50_1000.csv","knapPI_16_50_1000.csv"]

#Broadcast part a functions over files
# parta_exploratory_heatmaps.(files)
# parta_boxplots.(files)
temp=parta_KW_tests.(files)

function partc(file,fname="",points2=false)
    f_in(x)=SubString(x,8,length(x)-4)
    inst=f_in(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false)

    #Also read in constraint value
    con=open(file) do file
        fileText= readlines(file)
        con=fileText[3][3:end]
        return parse(Int,con)
    end

    #Create variables, knapsack options (n), weights (w), and values (v)
    n=size(data)[1]
    w=data[:,3]
    v=data[:,2]

    #Create objective function, which defaults to 0 if it violates the constraint,
    #and the sum is set to negative as this EA uses minimization
    # This constraint handling technique was the best for
    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : (sum(w.*x)-con))


    #Defined bit swapping function.
    #Selects two arbitrary genes and swaps their values.
    #Can experience "collisions" that don't change the
    #gene's output (i.e. swapping a 1 with a 1 or 0 with 0).
    function swap(recombinant::T) where {T <: BitVector}
        s = length(recombinant)
        p1=rand(1:s)
        p2=rand(1:s)
        while(p2==p1)
            p2=rand(1:s)
        end
        t1=copy(recombinant[p1])
        t2=copy(recombinant[p2])
        recombinant[p1]=t2
        recombinant[p2]=t1
        return recombinant
    end

    popsize=4*n;

    if points2
        points=mapreduce(y->map(x->(x,y),LinRange(1/(popsize*n),1/popsize,9)),vcat,0.1:0.1:0.9);
    else
        points=mapreduce(y->map(x->(x,y),LinRange(4/(popsize*n),4/popsize,9)),vcat,0.1:0.1:0.9);
    end

    function stats(l)
        temp=map(y->mapreduce(x->hcat(-minimum(x),Evolutionary.iterations(x),Evolutionary.f_calls(x)),vcat,l[y]),1:30)
        fit=map(y->map(x->temp[x][y],1:30),1:81)
        calls=map(y->map(x->temp[x][y],1:30),163:243)
        return fit, calls
    end

    #gafs=GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=points[8][1],crossoverRate=points[40][2],ɛ = 0.1);
    #rfs = [Evolutionary.optimize(f, x0(n), gafs, Evolutionary.Options(iterations=10000, successive_f_tol=10)) for _ in 1:30];
    gafs=map(x->GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    rfs = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafs) for _ in 1:30];
    #fs_fitness, fs_calls = (-minimum.(rfs), Evolutionary.f_calls.(rfs));
    fs_fitness, fs_calls = stats(rfs);

    gaswu=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    rswu = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gaswu) for _ in 1:30];
    swu_fitness, swu_calls = stats(rswu);

    gasws=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    rsws = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gasws) for _ in 1:30];
    sws_fitness, sws_calls = stats(rsws);

    fits=[fs_fitness,sws_fitness,swu_fitness]
    calls=[fs_calls,sws_calls,swu_calls]

    xs = [@sprintf("%.1E",i) for i = LinRange(1/(popsize*n),1/popsize,9)]
    ys = [string("C", i) for i = 0.1:0.1:0.9]

    #Adjusted plotter functions to plot 1 on top and 2 on bottom.
    function f_plotter(data,fname,instance)
        l= @layout [a ; b c]
        plot(data..., layout=l, tickfont=(3), titlefont=(8), plot_title=string(fname," Scores for", " ", instance))
    end

    function c_plotter(data,fname,instance)
        l= @layout [a ; b c]
        plot(data..., layout=l, tickfont=(3), titlefont=(8), plot_title=string(fname," Calls for", " ", instance))
    end
    #Create heatmaps for example plots, specifically to determine the best parameters for each setup across median, best, and worst performances.
    #Also investigate the most frequently occurring values. Applied to both the fitness and number of function calls.
    med_fit=map(x->hm(x,median),fits)
    med_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(median.(x)))),med_fit)
    mod_fit=map(x->hm(x,mode),fits)
    mod_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(mode.(x)))),mod_fit)
    max_fit=map(x->hm(x,maximum),fits)
    max_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(max.(x)))),max_fit)
    min_fit=map(x->hm(x,minimum),fits)
    min_fit_hm=map(x->heatmap(xs,ys,x,title=string("Best Score:", maximum(minimum.(x)))),min_fit)
    med_calls=map(x->hm(x,median),calls)
    med_calls_hm=map(x->heatmap(xs,ys,x),med_calls)
    mod_calls=map(x->hm(x,mode),calls)
    mod_calls_hm=map(x->heatmap(xs,ys,x),mod_calls)
    max_calls=map(x->hm(x,maximum),calls)
    max_calls_hm=map(x->heatmap(xs,ys,x),max_calls)
    min_calls=map(x->hm(x,minimum),calls)
    min_calls_hm=map(x->heatmap(xs,ys,x),min_calls)

    #Create output directory for figures
    if(fname!="")
        foldername=fname
    else
        foldername="figures"
    end
    # Create directory if it doesn't exist
    if !isdir(foldername)
        mkdir(foldername)
    end

    # Plot and save fitness heatmaps
    m1=f_plotter(med_fit_hm,"Median", string("Instance ", inst))
    m2=f_plotter(max_fit_hm,"Maximum", string("Instance ", inst))
    m3=f_plotter(mod_fit_hm,"Mode", string("Instance ", inst))
    m4=f_plotter(min_fit_hm,"Minimum", string("Instance ", inst))
    savefig(m1,string("./",foldername,"/Median_",inst))
    savefig(m2,string("./",foldername,"/Maximum_",inst))
    savefig(m3,string("./",foldername,"/Mode_",inst))
    savefig(m4,string("./",foldername,"/Minimum_",inst))

    # Plot and save calls heatmaps.
    m1=c_plotter(med_calls_hm,"Median", string("Instance ", inst))
    m2=c_plotter(max_calls_hm,"Maximum", string("Instance ", inst))
    m3=c_plotter(mod_calls_hm,"Mode", string("Instance ", inst))
    m4=c_plotter(min_calls_hm,"Minimum", string("Instance ", inst))
    savefig(m1,string("./",foldername,"/Median_calls_",inst))
    savefig(m2,string("./",foldername,"/Maximum_calls_",inst))
    savefig(m3,string("./",foldername,"/Mode_calls_",inst))
    savefig(m4,string("./",foldername,"/Minimum_calls_",inst))
end

partc.(files,false)


# We wanted to make boxplots for this, but we ran out of room in the report.
#function partc_boxplots(file)
#    f_in(x)=SubString(x,8,length(x)-4)
#    inst=f_in(file)
#    data=CSV.read(file,DataFrame,skipto=6,header=false)
#
#    #Also read in constraint value
#    con=open(file) do file
#        fileText= readlines(file)
#        con=fileText[3][3:end]
#        return parse(Int,con)
#    end
#
#    #Create variables, knapsack options (n), weights (w), and values (v)
#    n=size(data)[1]
#    w=data[:,3]
#    v=data[:,2]
#
#    #Best objective function and constraint method, penalty = how much we've violated the constraint.
#    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : (sum(w.*x)-con))
#
#    #Define different combinations of genetic algorithms with different types of mutation, and types of crossover
#    #uniform and singlepoint.
#    # Population size is pinned to 4*genome length, and mutation rate minimum/maximum are related to this value according to past work at purdue CITE.
#    # Crossover rates "average" tends to be 0.5, so we sampled from 0.1..0.9
#    popsize=4*n
#    points=mapreduce(y->map(x->(x,y),LinRange(1/(popsize*n),1/popsize,9)),vcat,0.1:0.1:0.9)
#    gafs=map(x->GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
#
#    #Run the optimization 30 times for each of the 81 combinations, collect distributions on fitness and calls, suppress output.
#    rfs = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafs) for _ in 1:30];
#    fs_fitness, fs_calls = stats(rfs);
#
#    #Define X and Y values for heatmap plots, with X corresponding to Mutation Rate, and Y to Crossover Rate
#    xs = [@sprintf("%.1E",i) for i = LinRange(1/(popsize*n),1/popsize,9)]
#    ys = [string("C", i) for i = 0.1:0.1:0.9]
#
#    # This and the following function allows easy plotting of the heatmaps, and manipulating the title values
#    ffmrm=reduce(hcat,map(x->reduce(vcat,fs_fitness[x:9:x+72]),1:9))
#    ffcrm=reduce(hcat,map(x->reduce(vcat,fs_fitness[x:x+8]),1:9:73))
#    fcmrm=reduce(hcat,map(x->reduce(vcat,fs_calls[x:9:x+72]),1:9))
#    fccrm=reduce(hcat,map(x->reduce(vcat,fs_calls[x:x+8]),1:9:73))
#
#    savefig(boxplot(permutedims(xs),ffmrm,leg=false, title="Box Plot of Fitness vs Mutation Rate"),string("./boxc_ss_flippedf_pen-con/Box_Fit_MR",inst))
#    savefig(boxplot(permutedims(ys),ffcrm,leg=false, title="Box Plot of Fitness vs Crossover Rate"),string("./boxc_ss_flippedf_pen-con/Box_Fit_CR",inst))
#    savefig(boxplot(permutedims(xs),fcmrm,leg=false, title="Box Plot of Function Calls vs Mutation Rate"),string("./boxc_ss_flippedf_pen-con/Box_Calls_MR",inst))
#    savefig(boxplot(permutedims(ys),fccrm,leg=false, title="Box Plot of Function Calls vs Crossover Rate"),string("./boxc_ss_flippedf_pen-con/Box_Calls_CR",inst))
#
#end
