def afford(shopping_items, budget):
    # TODO: Get most items for given budget
    d = shopping_items
    totcost = 0
    affordable={}
    thisbudget = budget
    
    while totcost<=thisbudget:
      if bool(d) is False:
        break
      item = min(d, key=d.get)
      itemcost = d.get(item)
      totcost = totcost + itemcost
      if totcost > thisbudget:
        break
      affordable.update({item:itemcost})
      
      del d[item]
      
    
    return affordable