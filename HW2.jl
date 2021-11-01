# To run this, you need to install Julia 1.6, and install the packages listed on the next line.
# This can be done by pressing "]" and typing add *package name*, then backspace to return to
# the julia REPL.
using Evolutionary, CSV, DataFrames, Distributions, HypothesisTests, Printf, Plots

#Set Filename and Read CSV into DataFrame
files=["knapPI_11_100_1000.csv","knapPI_13_200_1000.csv","knapPI_13_50_1000.csv","knapPI_14_50_1000.csv","knapPI_15_50_1000.csv","knapPI_16_50_1000.csv"]
# Define function for part A.
function parta_exploratory_heatmaps(file)
    f_in(x)=SubString(x,8,length(x)-4)
    inst=f_in(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false)

    #Also read in constraint value con=open(file) do file
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
    #f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : 0)

    #Create bitvector of zeros to represent an empty solution. These are randomized in the population initialization.
    #x0=flip(BitVector(zeros(n)))
    #x0=BitVector(rand(Bool,n))
    function x0(n)
        return flip(BitVector(zeros(n)))
    end

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
    ##gaswu=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    #gasws=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    # Function to return the fitness and function calls from each of the 30 replicates for each of the 81 combinations.
    function stats(l)
        temp=map(y->mapreduce(x->hcat(-minimum(x),Evolutionary.iterations(x),Evolutionary.f_calls(x)),vcat,l[y]),1:30)
        fit=map(y->map(x->temp[x][y],1:30),1:81)
        calls=map(y->map(x->temp[x][y],1:30),163:243)
        return fit, calls
    end

    # Check distribution difference with 98% Confidence Interval
    #function ks_pairwise(data)
    #    @suppress_err begin
    #    temp=map(y->map(x->ApproximateTwoSampleKSTest(data[x],data[y]).δ,y+1:y+8),1:72)
    #    #f(x)=(x<=0.02 ? x=1 : x=0)
    #    return temp #reshape(f.(temp),9,:)'
    #    end
    #end

    #Run the optimization 30 times for each of the 81 combinations, collect distributions on fitness and calls, suppress output.
    rfu = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafu) for _ in 1:30];
    fu_fitness, fu_calls = stats(rfu);

    rfs = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafs) for _ in 1:30];
    fs_fitness, fs_calls = stats(rfs);

    ruu = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gauu) for _ in 1:30];
    uu_fitness, uu_calls = stats(ruu);

    rus = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gaus) for _ in 1:30];
    us_fitness, us_calls = stats(rus);

    #rswu = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gaswu);
    #swus=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

    #rsws = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gasws);
    #swss=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

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

    # Plot and save fitness heatmaps
    m1=f_plotter(med_fit_hm,"Median", string("Instance ", inst))
    m2=f_plotter(max_fit_hm,"Maximum", string("Instance ", inst))
    m3=f_plotter(mod_fit_hm,"Mode", string("Instance ", inst))
    m4=f_plotter(min_fit_hm,"Minimum", string("Instance ", inst))
    savefig(m1,string("./figures/Median_",inst))
    savefig(m2,string("./figures/Maximum_",inst))
    savefig(m3,string("./figures/Mode_",inst))
    savefig(m4,string("./figures/Minimum_",inst))

    # Plot and save calls heatmaps.
    m1=c_plotter(med_calls_hm,"Median", string("Instance ", inst))
    m2=c_plotter(max_calls_hm,"Maximum", string("Instance ", inst))
    m3=c_plotter(mod_calls_hm,"Mode", string("Instance ", inst))
    m4=c_plotter(min_calls_hm,"Minimum", string("Instance ", inst))
    savefig(m1,string("./figures/Median_calls_",inst))
    savefig(m2,string("./figures/Maximum_calls_",inst))
    savefig(m3,string("./figures/Mode_calls_",inst))
    savefig(m4,string("./figures/Minimum_calls_",inst))

end

function parta_boxplots(file)
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
    #f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : 0)

    #Create bitvector of zeros to represent an empty solution. These are randomized in the population initialization.
    #x0=flip(BitVector(zeros(n)))
    #x0=BitVector(rand(Bool,n))
    function x0(n)
        return flip(BitVector(zeros(n)))
    end

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


    #Define different combinations of genetic algorithms with different types of mutation, and types of crossover
    #uniform and singlepoint.
    # Population size is pinned to 4*genome length, and mutation rate minimum/maximum are related to this value according to past work at purdue CITE.
    # Crossover rates "average" tends to be 0.5, so we sampled from 0.1..0.9
    popsize=4*n
    points=mapreduce(y->map(x->(x,y),LinRange(1/(popsize*n),1/popsize,9)),vcat,0.1:0.1:0.9)
    gafs=map(x->GA(populationSize=popsize,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    # Function to return the fitness and function calls from each of the 30 replicates for each of the 81 combinations.
    function stats(l)
        temp=map(y->mapreduce(x->hcat(-minimum(x),Evolutionary.iterations(x),Evolutionary.f_calls(x)),vcat,l[y]),1:30)
        fit=map(y->map(x->temp[x][y],1:30),1:81)
        calls=map(y->map(x->temp[x][y],1:30),163:243)
        return fit, calls
    end

    # Check distribution difference with 98% Confidence Interval
    #function ks_pairwise(data)
    #    @suppress_err begin
    #    temp=map(y->map(x->ApproximateTwoSampleKSTest(data[x],data[y]).δ,y+1:y+8),1:72)
    #    #f(x)=(x<=0.02 ? x=1 : x=0)
    #    return temp #reshape(f.(temp),9,:)'
    #    end
    #end

    #Run the optimization 30 times for each of the 81 combinations, collect distributions on fitness and calls, suppress output.
    rfs = [map(x->Evolutionary.optimize(f, x0(n), x, Evolutionary.Options(iterations=10000, successive_f_tol=10)),gafs) for _ in 1:30];
    fs_fitness, fs_calls = stats(rfs);

    ##gaswu=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    #gasws=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    #rswu = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gaswu);
    #swus=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

    #rsws = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gasws);
    #swss=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

    #Define X and Y values for heatmap plots, with X corresponding to Mutation Rate, and Y to Crossover Rate
    xs = [@sprintf("%.1E",i) for i = LinRange(1/(popsize*n),1/popsize,9)]
    ys = [string("C", i) for i = 0.1:0.1:0.9]

    # This and the following function allows easy plotting of the heatmaps, and manipulating the title values
    ffmrm=reduce(hcat,map(x->reduce(vcat,fs_fitness[x:9:x+72]),1:9))
    ffcrm=reduce(hcat,map(x->reduce(vcat,fs_fitness[x:x+8]),1:9:73))
    fcmrm=reduce(hcat,map(x->reduce(vcat,fs_calls[x:9:x+72]),1:9))
    fccrm=reduce(hcat,map(x->reduce(vcat,fs_calls[x:x+8]),1:9:73))

    savefig(boxplot(permutedims(xs),ffmrm,leg=false, title="Box Plot of Fitness vs Mutation Rate"),string("./box_ss_flippedf_pen-con/Box_Fit_MR",inst))
    savefig(boxplot(permutedims(ys),ffcrm,leg=false, title="Box Plot of Fitness vs Crossover Rate"),string("./box_ss_flippedf_pen-con/Box_Fit_CR",inst))
    savefig(boxplot(permutedims(xs),fcmrm,leg=false, title="Box Plot of Function Calls vs Mutation Rate"),string("./box_ss_flippedf_pen-con/Box_Calls_MR",inst))
    savefig(boxplot(permutedims(ys),fccrm,leg=false, title="Box Plot of Function Calls vs Crossover Rate"),string("./box_ss_flippedf_pen-con/Box_Calls_CR",inst))

end
#Broadcast part a over files
parta_boxplots.(files)
